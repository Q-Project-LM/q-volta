#pragma once
#include "../common.cuh"
#include "gemv.h"

namespace qv::cuda {
// q8 != nullptr: also emit the int8 GEMV input (fused quantization)
void rmsnorm(const float* x, const float* w, float* out, float* xs32, int n, float eps, cudaStream_t s,
             const Q8Buf* q8 = nullptr, int nt = 1);  // nt rows token-major (xs32 unused when nt > 1)
void embed(const QMat& E, const int* tok, float* out, cudaStream_t s);
void silu_mul(const float* g, const float* u, float* out, float* xs32, int n, cudaStream_t s, const Q8Buf* q8 = nullptr);
void argmax(const float* x, int n, int* out, cudaStream_t s);
void argmax_advance(const float* x, int n, int* tok, int* pos, int* gen, cudaStream_t s);

struct CandMeta {
    float mx;
    float sum_exp;
    int count;
};
void extract_top_candidates(const float* x, int n, float margin, int max_cand,
                            int* cand_id, float* cand_val, CandMeta* meta, cudaStream_t s);
void extract_top_candidates_batch(const float* x, int n, int n_rows, float margin, int max_cand,
                                  int* cand_id, float* cand_val, CandMeta* meta, cudaStream_t s);

constexpr int DRAFT_MAX_CAND = 256;
struct GpuDraftParams {
    float margin = 12.0f;
    float temperature = 0.8f;
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    int min_keep = 1;
    int max_extract = 4096;
};
void sample_draft_gpu(const float* logits, int n, const GpuDraftParams& p, uint64_t seed,
                      int* d_cand_id, float* d_cand_val, CandMeta* d_meta,
                      int* d_q_id, float* d_q_prob, int* d_q_cnt, int* d_draft_out, cudaStream_t s);
}  // namespace qv::cuda
