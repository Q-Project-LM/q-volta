#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

namespace qv::server {

struct PromptCheckpoint {
    std::vector<int> tokens;
    std::vector<uint8_t> blob;
    size_t bytes() const { return tokens.size() * sizeof(int) + blob.size(); }
};

struct ParkedSlot {
    int kv_tokens = 0;
    std::vector<uint8_t> kv_blob;
    std::vector<PromptCheckpoint> points;
    long long last_used = 0;

    size_t bytes() const {
        size_t total = kv_blob.size();
        for (const auto& pt : points) total += pt.bytes();
        return total;
    }
};

// Thread-safe LRU host-RAM pool of parked contexts bounded by slot count and total byte budget.
class ParkedCheckpointPool {
   public:
    struct Match {
        ParkedSlot slot;
        size_t point_idx = 0;
        size_t match_len = 0;
    };

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        slots_.clear();
        total_bytes_ = 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return slots_.size();
    }

    size_t total_bytes() const {
        std::lock_guard<std::mutex> lk(mu_);
        return total_bytes_;
    }

    // Ensures room for `need_bytes` (1 new slot) before device-to-host allocation so RAM usage does not spike 2x.
    bool reserve_for(size_t need_bytes, int max_slots, size_t max_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (max_slots <= 0 || max_bytes == 0 || need_bytes > max_bytes) return false;
        evict_locked((size_t)(max_slots - 1), max_bytes - need_bytes);
        return true;
    }

    bool insert(ParkedSlot slot, int max_slots, size_t max_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t sz = slot.bytes();
        if (max_slots <= 0 || max_bytes == 0 || sz > max_bytes) return false;
        evict_locked((size_t)(max_slots - 1), max_bytes - sz);
        slot.last_used = ++clock_;
        slots_.push_back(std::move(slot));
        total_bytes_ += sz;
        return true;
    }

    // Finds and removes the parked slot with the longest prefix strictly exceeding `min_len`.
    bool take_best_match(const std::vector<int>& prompt, size_t min_len, Match& out) {
        std::lock_guard<std::mutex> lk(mu_);
        int best_si = -1;
        size_t best_pi = 0;
        size_t best_len = min_len;
        long long best_used = -1;
        for (int si = 0; si < (int)slots_.size(); ++si) {
            const ParkedSlot& s = slots_[si];
            for (size_t pi = 0; pi < s.points.size(); ++pi) {
                const auto& toks = s.points[pi].tokens;
                if (toks.empty() || toks.size() <= min_len || toks.size() > prompt.size()) continue;
                if (toks.size() < best_len) continue;
                if (toks.size() == best_len && s.last_used <= best_used) continue;
                if (std::equal(toks.begin(), toks.end(), prompt.begin())) {
                    best_si = si;
                    best_pi = pi;
                    best_len = toks.size();
                    best_used = s.last_used;
                }
            }
        }
        if (best_si < 0) return false;
        out.slot = std::move(slots_[best_si]);
        out.point_idx = best_pi;
        out.match_len = best_len;
        total_bytes_ -= out.slot.bytes();
        slots_.erase(slots_.begin() + best_si);
        return true;
    }

   private:
    void evict_locked(size_t target_max_slots, size_t target_max_bytes) {
        while (!slots_.empty() && (slots_.size() > target_max_slots || total_bytes_ > target_max_bytes)) {
            size_t lru = 0;
            for (size_t i = 1; i < slots_.size(); ++i) {
                if (slots_[i].last_used < slots_[lru].last_used) lru = i;
            }
            total_bytes_ -= slots_[lru].bytes();
            slots_.erase(slots_.begin() + (std::ptrdiff_t)lru);
        }
    }

    mutable std::mutex mu_;
    std::vector<ParkedSlot> slots_;
    size_t total_bytes_ = 0;
    long long clock_ = 0;
};

}  // namespace qv::server
