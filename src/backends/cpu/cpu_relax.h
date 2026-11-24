// Portable spin-wait hint (x86 PAUSE / ARM YIELD).
#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define QV_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#include "avx2_neon.h"
#define QV_CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#define QV_CPU_RELAX() ((void)0)
#endif

// fp16 <-> fp32 scalar helpers (F16C on x86, native __fp16 on ARM)
#include <cstdint>
#include <cstring>
#if defined(__aarch64__)
inline float qv_h2f(uint16_t h) { __fp16 f; std::memcpy(&f, &h, 2); return (float)f; }
inline uint16_t qv_f2h(float v) { __fp16 f = (__fp16)v; uint16_t h; std::memcpy(&h, &f, 2); return h; }
#else
inline float qv_h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t qv_f2h(float v) { return _cvtss_sh(v, 0); }
#endif
