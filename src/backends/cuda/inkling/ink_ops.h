// Inkling-2-Mini (inkling2_loop) decode-step kernels (CUDA, sm_70). See docs/INKLING_ENGINE.md.
// All activations are fp32 [H]; GEMV inputs are int8-quantized (Q8Buf, canonical chunk permutation of gemv_q8).
#pragma once
#include "../common.cuh"
#include "../kernels/gemv.h"

namespace qv::cuda::ink {

// ---- int4 KV rows (global layers): per token and kv head, 128 channels -> 64 B of nibbles (byte b = channels 2b | 2b+1 << 4),
// plus one (scale, min) fp16 pair per 64-channel group, separately for K and V (K 64+8 B, V 64+8 B = 144 B per head-token).
// Called by all 128 threads of a block (thread c owns channel c). sh: >= 8 floats of shared memory.
__device__ __forceinline__ void kv4_store(float val, int c, uint8_t* qrow, __half2* srow, float* sh) {
    float mn = val, mx = val;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
    const int w = c >> 5;
    __syncthreads();
    if ((c & 31) == 0) { sh[w * 2] = mn; sh[w * 2 + 1] = mx; }
    __syncthreads();
    const int g = c >> 6;
    mn = fminf(sh[g * 4], sh[g * 4 + 2]);
    mx = fmaxf(sh[g * 4 + 1], sh[g * 4 + 3]);
    const __half sh_ = __float2half(fmaxf((mx - mn) * (1.f / 15.f), 1e-5f)), mh = __float2half(mn);
    const float sc = __half2float(sh_), mf = __half2float(mh);
    int q = __float2int_rn((val - mf) / sc);
    q = max(0, min(15, q));
    const int qn = __shfl_xor_sync(0xffffffffu, q, 1);
    if ((c & 1) == 0) qrow[c >> 1] = (uint8_t)(q | (qn << 4));
    if ((c & 63) == 0) srow[g] = __halves2half2(sh_, mh);
}

constexpr int H = 2304, NH = 24, NKV = 8, HD = 128, FF = 1152, NE = 128, TOPK = 8, KW = 4, REL = 16, REXT = 1024;
constexpr int HC = H / 32;  // 72 chunks of 32

// ---- embedding + final -------------------------------------------------------------------------------------------
// h = rms_w(dequant(embed[tok]), w); also copies h to blk0.
void embed_norm(const QMat& E, const int* tok, const float* w, float* h, float* blk0, cudaStream_t s, int T = 1);
void softcap_inplace(float* x, int n, float cap, cudaStream_t s);

// ---- attention residual (+ optional fused sconv/residual of the attention branch, norm and int8 quantization) ----------
struct ArArgs {
    int mode = 0;       // 0: partial = h ; 1: partial = post-attention residual (sconv(araw) [+ h])
    int B = 0;          // number of stored block states before this layer
    int Q = 1;          // number of queries
    int mix = 1;        // 0: out = partial (no attention residual; layer 0)
    int epi = 0;        // 0: write raw mix to raw_out ; 1: rms_w (nw[q]) -> xn_out + q8 (stream q) ; 2: rms_w -> xh_out (fp16)
    int ntok = 1, tok0 = 0, allpar = 0;  // prefill: ntok rows per buffer, allpar: token = blockIdx.x (else tok0)
    __half* xh_out = nullptr;            // epi 2: fp16 normalised rows [q][ntok][H]
    int rowy = 0;       // MTP verify: token row = blockIdx.y (grid (nblocks, rows)) instead of tok0/blockIdx.x
    int boundary = 0;   // mode 1: layer % 6 == 0: residual = a (no h), and h is appended to blocks[B]
    const float* h = nullptr;
    const float* araw = nullptr;
    const float* cw = nullptr;  // [H][4] attn_sconv weight
    float* cst = nullptr;       // [3][H] state
    float* res_out = nullptr;
    float* blocks = nullptr;    // [4][H]
    const float* queries = nullptr;
    long qbs = 0, rbs = 0;      // per-blockIdx.x strides (messages: one block per loop layer)
    float* raw_out = nullptr;
    const float* nw[4] = {nullptr, nullptr, nullptr, nullptr};
    float* xn_out = nullptr;
    Q8Buf q8;                   // stream q at q + q*H/4, d/sum/qsum + q*HC
    // fused k_finish (mode 0, one row, one block only): h = fin_res + (sconv(fin_y) + fin_y) / sqrt(5) is computed in place of reading h,
    // written to fin_h_out (and to fin_copy when non-null) and the MLP conv state fin_cst is advanced. Same arithmetic as k_finish.
    const float* fin_res = nullptr;
    const float* fin_y = nullptr;
    const float* fin_cw = nullptr;
    float* fin_cst = nullptr;
    float* fin_h_out = nullptr;
    float* fin_copy = nullptr;
};
void attn_res(const ArArgs& a, int nblocks, cudaStream_t s, int rows = 1);

// out stream s = rms_w(in[s], nw[s]) -> xn_out + s*H, q8 stream s
struct NormArgs {
    int S = 1;
    const float* in[4] = {};
    const float* nw[4] = {};
    float* xn_out = nullptr;
    Q8Buf q8;
    int T = 1;                 // prefill: row stride (rows per stream); grid (Tg, S) when T > 1
    int Tg = 0;
    __half* xh_out = nullptr;  // fp16 output instead of xn_out/q8
};
void norm_q8(const NormArgs& a, cudaStream_t s);

// ---- loop message gates ------------------------------------------------------------------------------------------
struct GateArgs {
    int S = 1;
    const float* h[4] = {};
    const float* msg[4] = {};
    const __half* down[4] = {};
    const __half* forget[4] = {};
    const __half* gain[4] = {};
    float* g = nullptr;      // [S][64]
    float* out[4] = {};
    int R = 1;               // MTP verify: rows per stream (grid.y = S*R); row r uses h/msg/out + r*stride
    long hrs = 0, mrs = 0, ors = 0;
};
void gate(const GateArgs& a, cudaStream_t s);

// ---- multi-matrix GEMV (q8 quantized matrices and fp16 matrices with fp32 x) ---------------------------------------
struct GD {
    int kind = 0;  // 0: quantized (int8 x), 1: fp16 (fp32 x)
    int N = 0, K = 0, gshift = 6;
    const uint32_t* q = nullptr;
    const uint16_t* sm8 = nullptr;
    const __half* rs = nullptr;
    const __half* w16 = nullptr;
    float zlo = 0, zstep = 0;
    Q8Vec xq{};
    const float* xf = nullptr;
    float* y = nullptr;
    int nblk = 0;
    int nr = 1;                // MTP verify rows (mgemv_rows): row r reads xq chunks + r*xstride (q: *8 ints) / xf + r*xfs, writes y + r*ystride
    long xstride = 0, xfs = 0, ystride = 0;
};
struct GDs {
    int n = 0;
    GD d[8];
};
void mgemv(GDs& g, int bits, cudaStream_t s);
void mgemv_rows(GDs& g, int bits, int nr, cudaStream_t s);  // nr <= 8 activation rows share each weight load; per row bit-identical to mgemv  // fills nblk, launches; all kind-0 descriptors must have `bits` bits

// ---- attention ------------------------------------------------------------------------------------------------------
struct PrepArgs {
    const float *qraw, *kraw, *vraw, *relraw, *dwk, *dwv;   // GEMV outputs (dwk/dwv: global layers only)
    const float *qscale, *relscale, *relproj;               // [24,128], [24,16], [16,1024]
    const float *kw, *vw;                                   // local: [1024][4]; global: static_w [4][1024]
    float *kst, *vst;                                       // conv state [3][1024]
    int glob;
    const int* pos;
    float* qs;      // [24][128] out
    float* rl;      // [24][1024] out (relative bias by distance)
    __half *kc, *vc;
    uint8_t *kq = nullptr, *vq = nullptr; __half2 *ks = nullptr, *vs = nullptr;  // int4 KV (global layers): kq/vq [8][cap][64], ks/vs [8][cap][2]
    // MTP verify (rows > 1): row r is position *pos + r; raw GEMV outputs of row r at raw + r*prs; k/v pre-conv inputs of every row are
    // also stored in khist/vhist [rows][1024] (conv-state commit); local layers write K/V rows to ktail/vtail [8][rows_alloc][128]
    // instead of the ring (the ring must keep positions < pos0 until the accepted rows are committed).
    int rows = 1; long prs = 0; float *khist = nullptr, *vhist = nullptr; __half *ktail = nullptr, *vtail = nullptr; int tail_rows = 8;
    int cap;        // slots per kv head in the cache (ring size 1024 for local layers)
    int ring;
};
void attn_prep(const PrepArgs& a, cudaStream_t s);
void attn_prep_rows(const PrepArgs& a, cudaStream_t s);
constexpr int PART_STRIDE = 130;
struct AttnArgs {
    const float* qs; const float* rl; const __half *kc, *vc;
    int cap, ring, window;
    const uint8_t *kq = nullptr, *vq = nullptr; const __half2 *ks = nullptr, *vs = nullptr;
    const int* pos;
    float* part;     // [8*nsplit][3][130]
    int nsplit;
    int bidir = 0;  // MTP draft block: every query row sees keys up to *pos + 7 (bidirectional inside the block)
    int rows = 1; long ss_rs = 0, gate_rs = 0; int q8_rs = 96;  // verify rows: sink/gate row strides, int8 output row stride (chunks)
    const __half *ktail = nullptr, *vtail = nullptr; int tail_rows = 8;  // local layers: keys at positions >= *pos0 live here
    const float* sink; const float* gate; Q8Buf out;   // combine: sink [24], gate [3072] -> q8 of sigmoid(gate)*o
};
void attn_part(const AttnArgs& a, cudaStream_t s);
void attn_combine(const AttnArgs& a, cudaStream_t s);
void attn_part_rows(const AttnArgs& a, cudaStream_t s);
void attn_combine_rows(const AttnArgs& a, cudaStream_t s);

// ---- MoE --------------------------------------------------------------------------------------------------------------
// router: logits = W x (fp16 [128][H], fp32 x), top-8 on sigmoid(l)+bias, weights = sigmoid(l)/sum*route_scale. ticket: one int, zero at start.
void router(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts, int* ticket, float route_scale,
            cudaStream_t s);
void router_rows(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts, int* ticket, float route_scale, int rows,
                 cudaStream_t s);  // logits [rows][128], ids/wts [rows][8], ticket [rows], x row stride H
struct ExpertW {  // stacked [E][N][K] bit-plane tensor
    const uint32_t* q; const uint16_t* sm8; const __half* rs;
    int N, K, bits, gshift;
    float zlo, zstep;
};
// act = softcap(g,4) sigmoid(g) softcap(u,25) for interleaved rows; output int8-quantized per slot (288 chunks total)
void moe_w13(const ExpertW& W, const Q8Vec& x, const int* ids, Q8Buf act, float gcap, float ucap, cudaStream_t s);
void moe_w2(const ExpertW& W, const Q8Vec& act, const int* ids, const float* wts, float* y, cudaStream_t s);
// ---- MTP verify (rows <= 8): union of the rows' experts -> grouped GEMVs that read each distinct expert once per matrix
void moe_union(const int* ids, int rows, int* uni_e, int* uni_cnt, int* uni_pair, int* uni_n, cudaStream_t s);
void moe_w13_rows(const ExpertW& W, const Q8Vec& x, const int* uni_e, const int* uni_cnt, const int* uni_pair, const int* uni_n, int max_u, Q8Buf act,
                  float gcap, float ucap, cudaStream_t s);  // x: row r at chunk r*36*2 ... (chunk stride K/32 per row); act chunk (row*8+slot)*(N/64)+rb
void moe_w2_rows(const ExpertW& W, const Q8Vec& act, const int* uni_e, const int* uni_cnt, const int* uni_pair, const int* uni_n, int max_u, float* ys,
                 const float* wts, float* y, int rows, cudaStream_t s);
void conv_commit_n(const float* hist, int rs, float* st, int D, int apps, const int* n_commit, cudaStream_t s);
void ring_commit(__half* kring, __half* vring, const __half* ktail, const __half* vtail, int tail_rows, int nslots, int rows, const int* pos0,
                 const int* n_commit, cudaStream_t s);
// h_out = res + sconv(y)
void finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, cudaStream_t s, int T = 1);
void conv_commit(const float* rows, long rs, int T, int D, float* st, cudaStream_t s);


// ---- MTP drafter kernels (docs/INKLING_MTP.md) ----------------------------------------------------------------------------
// out[row] = rms_w(sum_t alpha[t] * feat[row][t], nw): feat [rows][4][H]; writes the int8 stream `row` of q8 and (if non-null) fp32 out [rows][H].
// last_commit != null: a single row (*last_commit - 1) is mixed (the recurrent head's f_rnn) -> out [H].
void mix_q8(const float* feat, int rows, const float* alpha, const float* nw, Q8Buf q8, float* out, const int* last_commit, cudaStream_t s);
// context K/V of rows: k = rope(rms_w(kraw, knw)), v = vraw -> cache [8][cap][128] fp16 at position pos0 + row (ring when ring); rows >= *ncommit skipped
void ctx_post(const float* kraw, const float* vraw, const float* knw, __half* kc, __half* vc, int cap, int ring, int pos0_val, const int* pos0_ptr,
              const int* ncommit, int rows, cudaStream_t s);
// q (24 heads) / k (8 heads) of the 8 draft rows: rms_w + rope; q -> qs [8][3072] (x 128^-0.5), k and v -> tails [8][8][128] fp16
void draft_qk(const float* qraw, const float* kraw, const float* vraw, const float* qnw, const float* knw, const int* pos0, float* qs, __half* ktail,
              __half* vtail, cudaStream_t s);
void draft_combine(const AttnArgs& a, cudaStream_t s);  // a.part, nsplit, rows, out (int8 rows of 96 chunks)
void add_rows(float* x, const float* y, long n, cudaStream_t s);
void silu_mul_q8(const float* g, const float* u, int rows, int n, Q8Buf out, cudaStream_t s);  // out stream row: K = n
void fill_mask_rows(float* x, const float* mask, int rows, cudaStream_t s);                     // rows 1..rows-1 = mask
void mk_inputs(const float* xn, const float* f_rnn, float* xcat, int rows, cudaStream_t s);        // xcat[j] = [xn[j+1], f_rnn], j < rows
// one recurrence step j: raw = [wa | ctx_j + bias] ; state/latent update; latent -> q8 (8 chunks). prev token from *prev
void mk_step(const QMat& w1, const __half* wa, const float* ctxterm_j, const float* bias, const int* prev, float* state, Q8Buf q8lat, int first,
             cudaStream_t s);
// argmax over v = base[row] + bias + fold_bias; writes *tok (lowest index wins ties); scratch >= 2*ARGB floats/ints
void add_argmax(const float* base, const float* bias, const float* fold_bias, int n, float* sv, int* si, int* tok, cudaStream_t s);
// sampled draft (lossless speculative sampling): out[i] = cap * tanh((base + bias + fold_bias) / cap)
void draft_logits(const float* base, const float* bias, const float* fold_bias, int n, float cap, float* out, cudaStream_t s);
// sample a draft token from the filtered distribution of `logits` (top_k / top_p / min_p / temperature as in the request) with the seed read from *seed_ptr (device)
// mixed with `step`; writes the distribution (up to 256 candidates, descending) to q_id/q_prob/q_cnt and the token to *tok. cand_* scratch: max_extract entries.
struct DraftSampleParams {
    float margin = 8.0f, temperature = 1.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0, min_keep = 1, max_extract = 4096;
    bool operator==(const DraftSampleParams& o) const {
        return margin == o.margin && temperature == o.temperature && top_p == o.top_p && min_p == o.min_p && top_k == o.top_k && min_keep == o.min_keep && max_extract == o.max_extract;
    }
};
void draft_sample(const float* logits, int n, const DraftSampleParams& p, const unsigned long long* seed_ptr, int step, int* cand_id, float* cand_val, void* cand_meta,
                  int* q_id, float* q_prob, int* q_cnt, int* tok, cudaStream_t s);
void accept_rows(const int* vtok, const int* vy, int R, int* st, int* gen, int* nacc_hist, cudaStream_t s);
void advance_pos(int* st, cudaStream_t s);
void embed_row(const float* x, float* dst, int n, cudaStream_t s);

}  // namespace qv::cuda::ink
