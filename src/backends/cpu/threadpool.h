// Spin-barrier thread pool: pinned workers, static partitioning, no condvar / futex on the hot path.
// The calling thread is worker 0. Workers busy-wait on a generation counter (pause loop) and fall back to short
// sleeps only after a long idle period so an idle pool does not burn CPUs forever.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>
#include "cpu_relax.h"
#include <time.h>
#ifdef __linux__
#include <sched.h>
#define QV_HAVE_AFFINITY 1
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace qv::cpu {

class ThreadPool {
public:
    using Fn = void (*)(void* ctx, size_t begin, size_t end, int tid);

    // nthreads includes the caller. cpus: cpu ids to pin to (tid i -> cpus[i]); empty = first nthreads allowed cpus.
    explicit ThreadPool(int nthreads, std::vector<int> cpus = {}) : n_(nthreads < 1 ? 1 : nthreads) {
#ifdef QV_HAVE_AFFINITY
        sched_getaffinity(0, sizeof saved_, &saved_);  // restored in the destructor (caller is pinned meanwhile)
        if (cpus.empty())
            for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &saved_)) cpus.push_back(c);
#endif
        if (cpus.empty()) cpus.push_back(0);
        cpus_ = cpus;
        pin(cpus_[0 % cpus_.size()]);
        slots_ = new Slot[n_];
        for (int t = 1; t < n_; ++t)
            th_.emplace_back([this, t] { worker(t); });
    }
    ~ThreadPool() {
        stop_.store(true, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : th_) t.join();
        delete[] slots_;
#ifdef QV_HAVE_AFFINITY
        sched_setaffinity(0, sizeof saved_, &saved_);
#endif
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    int size() const { return n_; }

    // Static partition of [0,n) into size() contiguous ranges whose boundaries are multiples of `grain`.
    // fn(begin,end,tid) is called once per thread (possibly with an empty range skipped).
    template <class F>
    void parallel_for(size_t n, F&& fn, size_t grain = 1) {
        auto thunk = [](void* c, size_t b, size_t e, int t) { (*static_cast<std::remove_reference_t<F>*>(c))(b, e, t); };
        run(thunk, &fn, n, grain);
    }

    // Dynamic variant: [0,n) is cut into chunks of `chunk` items (a multiple of the caller's alignment need) that
    // threads claim from an atomic counter. A thread that is descheduled by the OS only costs the chunk it holds, not
    // the whole barrier (static partitioning stalls the barrier for a full scheduler slice, ~4 ms, whenever any one of
    // the pinned threads is preempted -- measured on this host at 63/64 threads).
    template <class F>
    void parallel_for_dyn(size_t n, F&& fn, size_t chunk) {
        auto thunk = [](void* c, size_t b, size_t e, int t) { (*static_cast<std::remove_reference_t<F>*>(c))(b, e, t); };
        run_dyn(thunk, &fn, n, chunk ? chunk : 1);
    }

private:
    struct alignas(64) Slot { std::atomic<uint32_t> done{0}; char pad[60]; };

    void pin(int cpu) {
#ifdef QV_HAVE_AFFINITY
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
        sched_setaffinity(0, sizeof s, &s);
#elif defined(__APPLE__)
        // macOS has no thread affinity: ask for the highest QoS so the scheduler keeps spinning workers on P-cores
        (void)cpu;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
        (void)cpu;
#endif
    }
    void range(int tid, size_t& b, size_t& e) const {
        const size_t units = (n_job_ + grain_ - 1) / grain_;
        const size_t per = units / n_, rem = units % n_;
        const size_t u0 = tid * per + (tid < (int)rem ? tid : rem);
        const size_t u1 = u0 + per + (tid < (int)rem ? 1 : 0);
        b = u0 * grain_; e = u1 * grain_;
        if (e > n_job_) e = n_job_;
        if (b > e) b = e;
    }
    void run(Fn fn, void* ctx, size_t n, size_t grain) {
        if (n_ == 1) { fn(ctx, 0, n, 0); return; }
        fn_ = fn; ctx_ = ctx; n_job_ = n; grain_ = grain ? grain : 1;
        const uint32_t g = (gen_.load(std::memory_order_relaxed) & ~1u) + 2;  // bit 0 = 0: static job
        gen_.store(g, std::memory_order_release);  // publishes fn_/ctx_/n_job_
        size_t b, e; range(0, b, e);
        if (b < e) fn(ctx, b, e, 0);
        for (int t = 1; t < n_; ++t)
            while (slots_[t].done.load(std::memory_order_acquire) != g) QV_CPU_RELAX();
    }
    void claim_loop(int tid) {
        for (;;) {
            uint64_t c = claim_.load(std::memory_order_acquire);
            const size_t idx = (size_t)(c & 0xffffffffu);
            if (idx >= units_) return;
            if (!claim_.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel)) continue;
            const size_t b = idx * grain_;
            size_t e = b + grain_; if (e > n_job_) e = n_job_;
            fn_(ctx_, b, e, tid);
            completed_.fetch_add(1, std::memory_order_release);
        }
    }
    void run_dyn(Fn fn, void* ctx, size_t n, size_t chunk) {
        if (n_ == 1 || n <= chunk) { fn(ctx, 0, n, 0); return; }
        fn_ = fn; ctx_ = ctx; n_job_ = n; grain_ = chunk;
        units_ = (n + chunk - 1) / chunk;
        completed_.store(0, std::memory_order_relaxed);
        const uint32_t g = ((gen_.load(std::memory_order_relaxed) & ~1u) + 2) | 1u;  // bit 0 = 1: dynamic job
        claim_.store((uint64_t)(g >> 1) << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_release);
        claim_loop(0);
        while (completed_.load(std::memory_order_acquire) != units_) QV_CPU_RELAX();
        claim_.store(~0ull, std::memory_order_release);  // retire: idx >= units for any late arrival
    }
    void worker(int tid) {
        pin(cpus_[tid % cpus_.size()]);
        uint32_t seen = 0;
        uint64_t spins = 0;
        for (;;) {
            uint32_t g;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                QV_CPU_RELAX();
                if (++spins > (1u << 22)) {  // ~ tens of ms idle: back off
                    timespec ts{0, 50000}; nanosleep(&ts, nullptr);
                }
            }
            spins = 0;
            if (stop_.load(std::memory_order_relaxed)) return;
            seen = g;
            if (g & 1u) { claim_loop(tid); continue; }  // dynamic job; late arrivals find the claim counter exhausted
            size_t b, e; range(tid, b, e);
            if (b < e) fn_(ctx_, b, e, tid);
            slots_[tid].done.store(g, std::memory_order_release);
        }
    }

#ifdef QV_HAVE_AFFINITY
    cpu_set_t saved_;
#endif
    int n_;
    std::vector<int> cpus_;
    std::vector<std::thread> th_;
    Slot* slots_ = nullptr;
    alignas(64) std::atomic<uint32_t> gen_{0};
    alignas(64) Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    size_t n_job_ = 0, grain_ = 1, units_ = 0;
    alignas(64) std::atomic<uint64_t> claim_{~0ull};
    alignas(64) std::atomic<size_t> completed_{0};
    std::atomic<bool> stop_{false};
};

}  // namespace qv::cpu
