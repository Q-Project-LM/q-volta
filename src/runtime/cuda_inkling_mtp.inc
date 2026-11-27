// MTP (DSpark drafter) part of InklingEngine; #included INSIDE the class body of cuda_inkling_engine.cu. Spec: docs/INKLING_MTP.md.
//
// One speculative cycle (one CUDA graph per verify width R = depth+1, R in {2,3,4,8}):
//   draft: anchor token + 7 mask rows -> 3 dense layers over the drafter's context K/V -> base logits + recurrent Markov head -> 7 greedy drafts
//   verify: R rows [anchor, d1..d_{R-1}] through the whole main model, row r bit-identical to the 1-token decode step at position p+r
//   accept: n = longest prefix with d_{i+1} == argmax(row i); tokens d1..dn and argmax(row n) are emitted; rows 0..n are committed
//   commit: conv states / local ring K,V of rows 0..n, then the drafter's context K/V of the committed rows from the captured features.
static constexpr int RR = 8;  // rows allocated for verify / draft

struct DLayer {
    QMat q, k, v, o, kc, vc, gate, up, down;
    const float *in_norm, *post_norm, *dflare, *qn, *kn, *knc;
};
bool mtp_ = false;
uint8_t* mw_ = nullptr;
size_t mw_bytes_ = 0;
DLayer dl_[3];
QMat w1_, foldw_;
const __half *rnn_wa_ = nullptr, *rnn_wb_ = nullptr;
const float *rnn_bias_ = nullptr, *mk_dflare_ = nullptr, *mtp_norm_ = nullptr, *mask_embed_ = nullptr, *fuse_alpha_ = nullptr, *fold_bias_ = nullptr;
// verify scratch (RR rows)
float *hv_, *prehv_, *blocksv_, *resv_, *uv_, *gsv_, *xnv_f_, *xmv_f_, *mrawv_, *mgv_, *msgsv_, *projv_, *qsv_, *rlv_, *partv_, *rlogv_, *wtsv_, *ggv_, *ysv_, *logitsv_,
    *khist_, *vhist_, *hist_a_, *hist_m_, *featv_;
int *idsv_, *ticketv_, *uni_e_, *uni_cnt_, *uni_pair_, *uni_n_, *vtok_, *vy_, *vst_, *nacc_hist_;
__half *ltk_, *ltv_;
Q8Buf q8xv_, q8mv_, q8actv_, q8ov_;
// draft scratch
float *xb_, *dq_, *dk_, *dv_, *dqs_, *dy_, *dgate_, *dup_, *dxn_, *dbase_, *dbias_, *dxcat_, *dctx_, *mkstate_, *f_rnn_, *zeros_rl_, *dpart_, *asv_, *blk_scratch_;
int* asi_;
__half *dktail_, *dvtail_, *dkc_[3], *dvc_[3];
Q8Buf dq8x_, dq8o_, dq8a_, dq8lat_;
float* featT_ = nullptr;  // prefill: [Tp_][4][H]
float* finT_ = nullptr;
int* d_dummy_i_ = nullptr;
int mtp_R_ = 8;
using ik_CandMeta = CandMeta;
static constexpr int SP_MAXC = 4096, SP_FAST = 256;
int* d_scid_ = nullptr; float* d_scval_ = nullptr; ik_CandMeta* d_smeta_ = nullptr;
int* h_scid_ = nullptr; float* h_scval_ = nullptr; ik_CandMeta* h_smeta_ = nullptr; int* h_svtok_ = nullptr; int* h_sst_ = nullptr;
std::vector<float> h_sfull_;
cudaGraphExec_t g_spA_[9] = {}, g_spB_[9] = {};
float spA_margin_[9] = {};
bool spA_sampled_[9] = {};
ik::DraftSampleParams spA_dsp_[9];
ik::DraftSampleParams dsp_;
bool dsp_on_ = false;  // draft_block(): sample the drafts instead of argmax (graph A with sampled drafts)
float* dlog_ = nullptr;
int *d_dq_id_ = nullptr, *d_dq_cnt_ = nullptr, *h_dq_id_ = nullptr, *h_dq_cnt_ = nullptr;
float *d_dq_prob_ = nullptr, *h_dq_prob_ = nullptr;
cudaGraphExec_t g_mtp_[9] = {};
MtpStats mstats_;

QMat mqmat(const ModelFile& mf, const std::string& n) {
    const TensorInfo& t = mf.at(n);
    if (t.kind != Kind::Quant || !t.meta8) throw std::runtime_error(n + " is not a meta8 quantized tensor");
    QMat m;
    m.sm8 = (const uint16_t*)(mw_ + t.off_sm);
    m.rs = (const __half*)(mw_ + t.off_rs);
    m.q = (const uint32_t*)(mw_ + t.off_q);
    m.zlo = -0.25f * (float)((1 << t.bits) - 1);
    m.zstep = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
    m.N = (int)t.shape[0];
    m.K = (int)t.shape[1];
    m.bits = t.bits;
    m.gshift = t.group == 128 ? 7 : 6;
    return m;
}
const float* mf32(const ModelFile& mf, const std::string& n) { return (const float*)(mw_ + mf.at(n).off); }
const __half* mf16(const ModelFile& mf, const std::string& n) { return (const __half*)(mw_ + mf.at(n).off); }

void load_mtp(const std::string& path) {
    ModelFile mf = ModelFile::open(path);
    mw_bytes_ = mf.bin_size;
    QV_CUDA_CHECK(cudaMalloc(&mw_, mw_bytes_));
    FILE* f = std::fopen(mf.bin_path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + mf.bin_path);
    const size_t CH = 64u << 20;
    void* pinned;
    QV_CUDA_CHECK(cudaMallocHost(&pinned, CH));
    size_t off = 0;
    while (off < mw_bytes_) {
        size_t n = std::fread(pinned, 1, std::min(CH, mw_bytes_ - off), f);
        if (n == 0) throw std::runtime_error("short read on mtp weights");
        QV_CUDA_CHECK(cudaMemcpy(mw_ + off, pinned, n, cudaMemcpyHostToDevice));
        off += n;
    }
    cudaFreeHost(pinned);
    std::fclose(f);
    for (int i = 0; i < 3; ++i) {
        const std::string P = "mtp.layers." + std::to_string(i) + ".";
        DLayer& d = dl_[i];
        d.q = mqmat(mf, P + "self_attn.q_proj.weight");
        d.k = mqmat(mf, P + "self_attn.k_proj.weight");
        d.v = mqmat(mf, P + "self_attn.v_proj.weight");
        d.o = mqmat(mf, P + "self_attn.o_proj.weight");
        d.kc = mqmat(mf, P + "self_attn.k_proj_ctx.weight");
        d.vc = mqmat(mf, P + "self_attn.v_proj_ctx.weight");
        d.gate = mqmat(mf, P + "mlp.gate_proj.weight");
        d.up = mqmat(mf, P + "mlp.up_proj.weight");
        d.down = mqmat(mf, P + "mlp.down_proj.weight");
        d.in_norm = mf32(mf, P + "input_layernorm.weight");
        d.post_norm = mf32(mf, P + "post_attention_layernorm.weight");
        d.dflare = mf32(mf, P + "dflare_norm.weight");
        d.qn = mf32(mf, P + "self_attn.q_norm.weight");
        d.kn = mf32(mf, P + "self_attn.k_norm.weight");
        d.knc = mf32(mf, P + "self_attn.k_norm_ctx.weight");
    }
    w1_ = mqmat(mf, "mtp.markov.w1");
    foldw_ = mqmat(mf, "mtp.markov.fold_weight");
    rnn_wa_ = mf16(mf, "mtp.markov.rnn_wa");
    rnn_wb_ = mf16(mf, "mtp.markov.rnn_wb");
    rnn_bias_ = mf32(mf, "mtp.markov.rnn_proj.bias");
    mk_dflare_ = mf32(mf, "mtp.markov.dflare_norm.weight");
    fold_bias_ = mf32(mf, "mtp.markov.fold_bias");
    mtp_norm_ = mf32(mf, "mtp.norm.weight");
    mask_embed_ = mf32(mf, "mtp.mask_embed");
    fuse_alpha_ = mf32(mf, "mtp.fuse_alpha");
}

void alloc_mtp() {
    const int H = ik::H, N = cfg_.ink.unpadded_vocab;
    hv_ = dalloc<float>(RR * H); prehv_ = dalloc<float>(RR * H); blocksv_ = dalloc<float>((size_t)4 * RR * H); resv_ = dalloc<float>(RR * H);
    uv_ = dalloc<float>(3 * RR * H); gsv_ = dalloc<float>(3 * RR * H); xnv_f_ = dalloc<float>(3 * RR * H); xmv_f_ = dalloc<float>(RR * H);
    mrawv_ = dalloc<float>(RR * H); mgv_ = dalloc<float>(RR * H); msgsv_ = dalloc<float>((size_t)12 * 4 * RR * H);
    projv_ = dalloc<float>((size_t)RR * 8704); qsv_ = dalloc<float>((size_t)RR * 3072); rlv_ = dalloc<float>((size_t)RR * 24 * 1024);
    partv_ = dalloc<float>((size_t)RR * ik::NKV * std::max(NSPLIT_L, NSPLIT_G) * 3 * ik::PART_STRIDE);
    rlogv_ = dalloc<float>(RR * 128); wtsv_ = dalloc<float>(RR * 8); ggv_ = dalloc<float>(4 * RR * 64); ysv_ = dalloc<float>((size_t)RR * 8 * H);
    logitsv_ = dalloc<float>((size_t)RR * N);
    khist_ = dalloc<float>((size_t)48 * RR * 1024); vhist_ = dalloc<float>((size_t)48 * RR * 1024);
    hist_a_ = dalloc<float>((size_t)48 * RR * H); hist_m_ = dalloc<float>((size_t)48 * RR * H);
    featv_ = dalloc<float>((size_t)RR * 4 * H);
    idsv_ = dalloc<int>(RR * 8); ticketv_ = dalloc<int>(RR); uni_e_ = dalloc<int>(64); uni_cnt_ = dalloc<int>(64); uni_pair_ = dalloc<int>(64 * 8); uni_n_ = dalloc<int>(1);
    vtok_ = dalloc<int>(RR); vy_ = dalloc<int>(RR); vst_ = dalloc<int>(8); nacc_hist_ = dalloc<int>(16);
    ltk_ = dalloc<__half>((size_t)n_lslots_ * ik::NKV * RR * ik::HD); ltv_ = dalloc<__half>((size_t)n_lslots_ * ik::NKV * RR * ik::HD);
    q8xv_ = q8alloc(3 * RR * ik::HC); q8mv_ = q8alloc(RR * ik::HC); q8actv_ = q8alloc(RR * 8 * 36); q8ov_ = q8alloc(RR * 96);
    xb_ = dalloc<float>(RR * H); dq_ = dalloc<float>((size_t)RR * 3072); dk_ = dalloc<float>(RR * 1024); dv_ = dalloc<float>(RR * 1024);
    dqs_ = dalloc<float>((size_t)RR * 3072); dy_ = dalloc<float>(RR * H); dgate_ = dalloc<float>((size_t)RR * 9216); dup_ = dalloc<float>((size_t)RR * 9216);
    dxn_ = dalloc<float>(RR * H); dbase_ = dalloc<float>((size_t)RR * N); dbias_ = dalloc<float>(N); dxcat_ = dalloc<float>(RR * 2 * H);
    dctx_ = dalloc<float>(RR * 768); mkstate_ = dalloc<float>(256); f_rnn_ = dalloc<float>(H); zeros_rl_ = dalloc<float>((size_t)RR * 24 * 1024);
    dpart_ = dalloc<float>((size_t)RR * ik::NKV * std::max(NSPLIT_L, NSPLIT_G) * 3 * ik::PART_STRIDE);
    asv_ = dalloc<float>(256); asi_ = dalloc<int>(256); blk_scratch_ = dalloc<float>(RR * H);
    dktail_ = dalloc<__half>((size_t)ik::NKV * RR * ik::HD); dvtail_ = dalloc<__half>((size_t)ik::NKV * RR * ik::HD);
    for (int i = 0; i < 3; ++i) {
        const bool loc = i == 0 || i == 2;
        const size_t cap = loc ? 1024 : opt_.max_ctx;
        dkc_[i] = dalloc<__half>((size_t)ik::NKV * cap * ik::HD);
        dvc_[i] = dalloc<__half>((size_t)ik::NKV * cap * ik::HD);
    }
    dq8x_ = q8alloc(RR * ik::HC); dq8o_ = q8alloc(RR * 96); dq8a_ = q8alloc(RR * 288); dq8lat_ = q8alloc(8);
    // lossless speculative sampling (T > 0): candidate extraction scratch
    d_scid_ = dalloc<int>((size_t)RR * SP_MAXC); d_scval_ = dalloc<float>((size_t)RR * SP_MAXC); d_smeta_ = dalloc<ik_CandMeta>(RR);
    QV_CUDA_CHECK(cudaMallocHost(&h_scid_, (size_t)RR * SP_MAXC * sizeof(int)));
    QV_CUDA_CHECK(cudaMallocHost(&h_scval_, (size_t)RR * SP_MAXC * sizeof(float)));
    QV_CUDA_CHECK(cudaMallocHost(&h_smeta_, RR * sizeof(ik_CandMeta)));
    QV_CUDA_CHECK(cudaMallocHost(&h_svtok_, RR * sizeof(int)));
    QV_CUDA_CHECK(cudaMallocHost(&h_sst_, 8 * sizeof(int)));
    std::memset(h_sst_, 0, 8 * sizeof(int));
    dlog_ = dalloc<float>(N); d_dq_id_ = dalloc<int>(8 * 256); d_dq_prob_ = dalloc<float>(8 * 256); d_dq_cnt_ = dalloc<int>(8);
    QV_CUDA_CHECK(cudaMallocHost(&h_dq_id_, 8 * 256 * sizeof(int)));
    QV_CUDA_CHECK(cudaMallocHost(&h_dq_prob_, 8 * 256 * sizeof(float)));
    QV_CUDA_CHECK(cudaMallocHost(&h_dq_cnt_, 8 * sizeof(int)));
    extract_top_candidates_batch(logitsv_, N, RR, 14.0f, SP_MAXC, d_scid_, d_scval_, d_smeta_, stream_);  // allocates the kernel's static scratch outside capture
    argmax_advance(logitsv_, N, vy_, nullptr, nullptr, stream_);  // same for the argmax scratch (the first graph capture may come before any plain step)
    QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---- drafter context K/V of `rows` rows of captured features (feat [rows][4][H]) at positions pos0 + r; rows >= *ncommit skipped
void mtp_ingest(const float* feat, int rows, int pos0_val, const int* pos0_ptr, const int* ncommit, const int* last_ptr) {
    cudaStream_t s = stream_;
    for (int i = 0; i < 3; ++i) {
        const DLayer& d = dl_[i];
        ik::mix_q8(feat, rows, fuse_alpha_ + i * 4, d.dflare, dq8x_, nullptr, nullptr, s);
        ik::GDs G;
        G.d[G.n++] = gd_rows(d.kc, dq8x_, dk_, 1024, ik::HC, rows);
        G.d[G.n++] = gd_rows(d.vc, dq8x_, dv_, 1024, ik::HC, rows);
        ik::mgemv_rows(G, d.kc.bits, rows, s);
        const bool loc = i != 1;
        ik::ctx_post(dk_, dv_, d.knc, dkc_[i], dvc_[i], loc ? 1024 : opt_.max_ctx, loc, pos0_val, pos0_ptr, ncommit, rows, s);
    }
    // recurrent head context: features of the last committed row
    const int* lp = last_ptr ? last_ptr : ncommit ? ncommit : (d_dummy_i_ + rows - 1);
    ik::mix_q8(feat, 1, fuse_alpha_ + 12, mk_dflare_, Q8Buf{}, f_rnn_, lp, s);
}
static ik::GD gd_rows(const QMat& W, const Q8Buf& x, float* y, long ystride, long xstride, int nr, int stream = 0, int rows_per_stream = RR) {
    ik::GD g;
    g.kind = 0; g.N = W.N; g.K = W.K; g.gshift = W.gshift; g.q = W.q; g.sm8 = W.sm8; g.rs = W.rs; g.zlo = W.zlo; g.zstep = W.zstep;
    const long cb = (long)stream * rows_per_stream * (W.K / 32 > 0 ? xstride : 0);
    g.xq = Q8Vec{x.q + cb * 8, x.d + cb, x.sum + cb, x.qsum + cb};
    g.y = y; g.nr = nr; g.xstride = xstride; g.ystride = ystride;
    return g;
}
static ik::GD gdh_rows(const DMat16& W, const float* x, long xfs, float* y, long ystride, int nr) {
    ik::GD g;
    g.kind = 1; g.N = W.N; g.K = W.K; g.w16 = W.w; g.xf = x; g.xfs = xfs; g.y = y; g.nr = nr; g.ystride = ystride;
    return g;
}

// ---- verify: R rows through all 48 applications, row r == plain decode step at position *vpos + r
void verify_forward(int R) {
    cudaStream_t s = stream_;
    const int H = ik::H;
    ik::embed_norm(embed_, vtok_, embed_norm_, hv_, blocksv_, s, R);
    for (const App& A : apps_) {
        run_app_v(A, R);
        if (A.idx == 4 || A.idx == 22 || A.idx == 40) {
            const int tap = A.idx == 4 ? 0 : A.idx == 22 ? 1 : 2;
            QV_CUDA_CHECK(cudaMemcpy2DAsync(featv_ + (size_t)tap * H, 4 * H * 4, hv_, H * 4, H * 4, R, cudaMemcpyDeviceToDevice, s));
        }
        if (A.L == cfg_.ink.loop_end - 1 && A.p < cfg_.ink.loop_passes - 1) {
            ik::ArArgs m;
            m.mode = 0; m.B = 1; m.Q = 4; m.mix = 1; m.epi = 0; m.ntok = RR; m.rowy = 1;
            m.h = hv_; m.blocks = blocksv_ + (size_t)2 * RR * H;
            m.queries = msgq_; m.qbs = 4 * H; m.raw_out = msgsv_; m.rbs = (long)4 * RR * H;
            ik::attn_res(m, 12, s, R);
        }
    }
    ik::ArArgs f;
    f.mode = 0; f.B = 4; f.Q = 1; f.mix = 1; f.epi = 1; f.ntok = RR; f.rowy = 1;
    f.h = hv_; f.blocks = blocksv_; f.queries = out_query_; f.nw[0] = final_norm_; f.xn_out = xnv_f_; f.q8 = q8xv_;
    ik::attn_res(f, 1, s, R);
    QV_CUDA_CHECK(cudaMemcpy2DAsync(featv_ + (size_t)3 * H, 4 * H * 4, xnv_f_, H * 4, H * 4, R, cudaMemcpyDeviceToDevice, s));
    const int nt = R <= 2 ? 2 : R == 3 ? 3 : R == 4 ? 4 : 8;
    const int N = unembed_.N;
    gemv_q8_multi(unembed_, Q8Vec(q8xv_), nt, logitsv_, nullptr, s);
    ik::softcap_inplace(logitsv_, R * N, cfg_.ink.final_logit_softcap, s);
    for (int r = 0; r < R; ++r) argmax_advance(logitsv_ + (size_t)r * N, cfg_.ink.unpadded_vocab, vy_ + r, nullptr, nullptr, s);
}

void run_app_v(const App& A, int R) {
    const InklingConfig& K = cfg_.ink;
    const LayerW& W = lw_[A.L];
    cudaStream_t s = stream_;
    const int H = ik::H, L = A.L, p = A.p, li = A.L - K.loop_start;
    const bool glob = W.glob, msg = p > 0;
    const float* hin = (L == K.loop_start) ? prehv_ : hv_;
    const int Battn = (L + 5) / 6;
    const int Qa = glob ? 3 : 1;
    const long P8 = (long)RR * H;
    float* araw = hist_a_ + (size_t)A.idx * RR * H;
    float* ymoe = hist_m_ + (size_t)A.idx * RR * H;
    const int* vpos = vst_ + 1;
    if (!msg) {
        ik::ArArgs a;
        a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = L > 0; a.epi = 1; a.ntok = RR; a.rowy = 1;
        a.h = hin; a.blocks = blocksv_; a.queries = W.attn_q;
        for (int q = 0; q < Qa; ++q) a.nw[q] = W.nw[q];
        a.xn_out = xnv_f_; a.q8 = q8xv_;
        ik::attn_res(a, 1, s, R);
    } else {
        ik::ArArgs a;
        a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = 1; a.epi = 0; a.ntok = RR; a.rowy = 1;
        a.h = hin; a.blocks = blocksv_; a.queries = W.attn_q; a.raw_out = uv_;
        ik::attn_res(a, 1, s, R);
        ik::GateArgs g;
        g.S = 3; g.g = ggv_; g.R = R; g.hrs = H; g.mrs = H; g.ors = H;
        for (int q = 0; q < 3; ++q) {
            g.h[q] = glob ? uv_ + q * P8 : uv_;
            g.msg[q] = msgsv_ + ((size_t)li * 4 + q) * P8;
            g.down[q] = W.ctrl[q].down; g.forget[q] = W.ctrl[q].forget; g.gain[q] = W.ctrl[q].gain;
            g.out[q] = gsv_ + q * P8;
        }
        ik::gate(g, s);
        ik::NormArgs n;
        n.S = 3; n.T = RR; n.Tg = R;
        for (int q = 0; q < 3; ++q) { n.in[q] = gsv_ + q * P8; n.nw[q] = W.nw[q]; }
        n.xn_out = xnv_f_; n.q8 = q8xv_;
        ik::norm_q8(n, s);
    }
    const int sk = (glob || msg) ? 1 : 0, sv = (glob || msg) ? 2 : 0;
    {
        ik::GDs G;
        G.d[G.n++] = gd_rows(W.wq, q8xv_, projv_ + 0, 8704, ik::HC, R, 0, RR);
        G.d[G.n++] = gd_rows(W.wk, q8xv_, projv_ + 3072, 8704, ik::HC, R, sk, RR);
        G.d[G.n++] = gd_rows(W.wv, q8xv_, projv_ + 4096, 8704, ik::HC, R, sv, RR);
        G.d[G.n++] = gd_rows(W.wg, q8xv_, projv_ + 5120, 8704, ik::HC, R, 0, RR);
        G.d[G.n++] = gdh_rows(W.wr, xnv_f_, H, projv_ + 8192, 8704, R);
        G.d[G.n++] = gdh_rows(W.sink, xnv_f_, H, projv_ + 8576, 8704, R);
        if (glob) {
            G.d[G.n++] = gdh_rows(W.kproj, xnv_f_ + (size_t)sk * P8, H, projv_ + 8600, 8704, R);
            G.d[G.n++] = gdh_rows(W.vproj, xnv_f_ + (size_t)sv * P8, H, projv_ + 8632, 8704, R);
        }
        ik::mgemv_rows(G, W.wq.bits, R, s);
    }
    const bool ring = !glob;
    const int cap = glob ? opt_.max_ctx : 1024;
    __half* kc = glob ? (kv4_ ? nullptr : gk_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD) : lk_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
    __half* vc = glob ? (kv4_ ? nullptr : gv_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD) : lv_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
    __half* ktl = glob ? nullptr : ltk_ + (size_t)A.kvslot * ik::NKV * RR * ik::HD;
    __half* vtl = glob ? nullptr : ltv_ + (size_t)A.kvslot * ik::NKV * RR * ik::HD;
    {
        ik::PrepArgs P;
        P.qraw = projv_; P.kraw = projv_ + 3072; P.vraw = projv_ + 4096; P.relraw = projv_ + 8192; P.dwk = projv_ + 8600; P.dwv = projv_ + 8632;
        P.qscale = W.qscale; P.relscale = W.relscale; P.relproj = W.relproj; P.kw = W.kw; P.vw = W.vw;
        P.kst = cst_k_ + (size_t)A.idx * 3 * 1024; P.vst = cst_v_ + (size_t)A.idx * 3 * 1024;
        P.glob = glob; P.pos = vpos; P.qs = qsv_; P.rl = rlv_; P.kc = kc; P.vc = vc; P.cap = cap; P.ring = ring;
        P.rows = R; P.prs = 8704; P.khist = khist_ + (size_t)A.idx * RR * 1024; P.vhist = vhist_ + (size_t)A.idx * RR * 1024;
        P.ktail = ktl; P.vtail = vtl; P.tail_rows = RR;
        if (glob && kv4_) { P.kq = gkq_ + grow(A) * 64; P.vq = gvq_ + grow(A) * 64; P.ks = gks_ + grow(A) * 2; P.vs = gvs_ + grow(A) * 2; }
        ik::attn_prep_rows(P, s);
    }
    {
        ik::AttnArgs T;
        T.qs = qsv_; T.rl = rlv_; T.kc = kc; T.vc = vc; T.cap = cap; T.ring = ring; T.window = glob ? 0 : K.window;
        if (glob && kv4_) { T.kq = gkq_ + grow(A) * 64; T.vq = gvq_ + grow(A) * 64; T.ks = gks_ + grow(A) * 2; T.vs = gvs_ + grow(A) * 2; }
        T.pos = vpos; T.part = partv_; T.nsplit = glob ? NSPLIT_G : NSPLIT_L;
        T.sink = projv_ + 8576; T.gate = projv_ + 5120; T.out = q8ov_;
        T.rows = R; T.ss_rs = 8704; T.gate_rs = 8704; T.q8_rs = 96; T.ktail = ktl; T.vtail = vtl; T.tail_rows = RR;
        ik::attn_part_rows(T, s);
        ik::attn_combine_rows(T, s);
    }
    {
        ik::GDs G;
        G.d[G.n++] = gd_rows(W.wo, q8ov_, araw, H, 96, R, 0, RR);
        ik::mgemv_rows(G, W.wo.bits, R, s);
    }
    {
        ik::ArArgs b;
        b.mode = 1; b.B = Battn; b.Q = 1; b.mix = 1; b.boundary = (L % 6 == 0); b.ntok = RR; b.rowy = 1;
        b.h = hin; b.araw = araw; b.cw = W.attn_sconv; b.cst = cst_a_ + (size_t)A.idx * 3 * H; b.res_out = resv_; b.blocks = blocksv_;
        b.queries = W.mlp_q;
        if (!msg) {
            b.epi = 1; b.nw[0] = W.mlp_norm; b.xn_out = xmv_f_; b.q8 = q8mv_;
            ik::attn_res(b, 1, s, R);
        } else {
            b.epi = 0; b.raw_out = mrawv_;
            ik::attn_res(b, 1, s, R);
            ik::GateArgs g;
            g.S = 1; g.g = ggv_; g.R = R; g.hrs = H; g.mrs = H; g.ors = H;
            g.h[0] = mrawv_; g.msg[0] = msgsv_ + ((size_t)li * 4 + 3) * P8;
            g.down[0] = W.ctrl[3].down; g.forget[0] = W.ctrl[3].forget; g.gain[0] = W.ctrl[3].gain; g.out[0] = mgv_;
            ik::gate(g, s);
            ik::NormArgs n;
            n.S = 1; n.T = RR; n.Tg = R; n.in[0] = mgv_; n.nw[0] = W.mlp_norm; n.xn_out = xmv_f_; n.q8 = q8mv_;
            ik::norm_q8(n, s);
        }
    }
    ik::router_rows(W.router.w, W.router_bias, xmv_f_, rlogv_, idsv_, wtsv_, ticketv_, K.route_scale, R, s);
    ik::moe_union(idsv_, R, uni_e_, uni_cnt_, uni_pair_, uni_n_, s);
    ik::moe_w13_rows(W.w13, Q8Vec(q8mv_), uni_e_, uni_cnt_, uni_pair_, uni_n_, 8 * R, q8actv_, K.gate_softcap, K.up_softcap, s);
    ik::moe_w2_rows(W.w2, Q8Vec(q8actv_), uni_e_, uni_cnt_, uni_pair_, uni_n_, 8 * R, ysv_, wtsv_, ymoe, R, s);
    float* hout = (L == K.loop_start - 1) ? prehv_ : hv_;
    ik::finish(resv_, ymoe, W.mlp_sconv, cst_m_ + (size_t)A.idx * 3 * H, hout, s, R);
}

// ---- draft block (7 greedy tokens into vtok_[1..7]); anchor = vst_[0], position vst_[1]
void draft_block(int nsteps = RR - 1) {  // nsteps = number of drafts needed (R - 1): the Markov steps are sequential, so fewer steps change nothing
    cudaStream_t s = stream_;
    const int H = ik::H, N = cfg_.ink.unpadded_vocab;
    const int* pos0 = vst_ + 1;
    ik::embed_norm(embed_, vst_, embed_norm_, xb_, blk_scratch_, s, 1);
    ik::fill_mask_rows(xb_, mask_embed_, RR, s);
    for (int i = 0; i < 3; ++i) {
        const DLayer& d = dl_[i];
        const bool loc = i != 1;
        ik::NormArgs n;
        n.S = 1; n.T = RR; n.Tg = RR; n.in[0] = xb_; n.nw[0] = d.in_norm; n.xn_out = dxn_; n.q8 = dq8x_;
        ik::norm_q8(n, s);
        {
            ik::GDs G;
            G.d[G.n++] = gd_rows(d.q, dq8x_, dq_, 3072, ik::HC, RR);
            G.d[G.n++] = gd_rows(d.k, dq8x_, dk_, 1024, ik::HC, RR);
            G.d[G.n++] = gd_rows(d.v, dq8x_, dv_, 1024, ik::HC, RR);
            ik::mgemv_rows(G, d.q.bits, RR, s);
        }
        ik::draft_qk(dq_, dk_, dv_, d.qn, d.kn, pos0, dqs_, dktail_, dvtail_, s);
        {
            ik::AttnArgs T;
            T.qs = dqs_; T.rl = zeros_rl_; T.kc = dkc_[i]; T.vc = dvc_[i]; T.cap = loc ? 1024 : opt_.max_ctx; T.ring = loc; T.window = loc ? 1024 + 8 : 0;
            T.pos = pos0; T.part = dpart_; T.nsplit = loc ? NSPLIT_L : NSPLIT_G; T.out = dq8o_;
            T.rows = RR; T.q8_rs = 96; T.ktail = dktail_; T.vtail = dvtail_; T.tail_rows = RR; T.bidir = 1;
            ik::attn_part_rows(T, s);
            ik::draft_combine(T, s);
        }
        {
            ik::GDs G;
            G.d[G.n++] = gd_rows(d.o, dq8o_, dy_, H, 96, RR);
            ik::mgemv_rows(G, d.o.bits, RR, s);
        }
        ik::add_rows(xb_, dy_, (long)RR * H, s);
        n.nw[0] = d.post_norm;
        ik::norm_q8(n, s);
        {
            ik::GDs G;
            G.d[G.n++] = gd_rows(d.gate, dq8x_, dgate_, 9216, ik::HC, RR);
            G.d[G.n++] = gd_rows(d.up, dq8x_, dup_, 9216, ik::HC, RR);
            ik::mgemv_rows(G, d.gate.bits, RR, s);
        }
        ik::silu_mul_q8(dgate_, dup_, RR, 9216, dq8a_, s);
        {
            ik::GDs G;
            G.d[G.n++] = gd_rows(d.down, dq8a_, dy_, H, 288, RR);
            ik::mgemv_rows(G, d.down.bits, RR, s);
        }
        ik::add_rows(xb_, dy_, (long)RR * H, s);
    }
    {
        ik::NormArgs n;
        n.S = 1; n.T = RR; n.Tg = RR; n.in[0] = xb_; n.nw[0] = mtp_norm_; n.xn_out = dxn_; n.q8 = dq8x_;
        ik::norm_q8(n, s);
    }
    gemv_q8_multi(unembed_, Q8Vec(dq8x_), 8, dbase_, nullptr, s);
    ik::mk_inputs(dxn_, f_rnn_, dxcat_, RR - 1, s);
    {
        ik::GDs G;
        G.d[G.n++] = gdh_rows(DMat16{rnn_wb_, 768, 4608}, dxcat_, 2 * H, dctx_, 768, RR - 1);
        ik::mgemv_rows(G, 5, RR - 1, s);
    }
    for (int j = 0; j < nsteps; ++j) {
        ik::mk_step(w1_, rnn_wa_, dctx_ + (size_t)j * 768, rnn_bias_, j == 0 ? vst_ : vtok_ + j, mkstate_, dq8lat_, j == 0, s);
        gemv_q8(foldw_, Q8Vec(dq8lat_), dbias_, nullptr, s);
        if (!dsp_on_) {
            ik::add_argmax(dbase_ + (size_t)(j + 1) * N, dbias_, fold_bias_, N, asv_, asi_, vtok_ + j + 1, s);
        } else {
            ik::draft_logits(dbase_ + (size_t)(j + 1) * N, dbias_, fold_bias_, N, cfg_.ink.final_logit_softcap, dlog_, s);
            ik::draft_sample(dlog_, N, dsp_, reinterpret_cast<const unsigned long long*>(vst_ + 6), j, d_scid_, d_scval_, d_smeta_, d_dq_id_ + j * 256, d_dq_prob_ + j * 256,
                             d_dq_cnt_ + j, vtok_ + j + 1, s);
        }
    }
}

void mtp_cycle(int R) {
    cudaStream_t s = stream_;
    draft_block(R - 1);
    QV_CUDA_CHECK(cudaMemcpyAsync(vtok_, vst_, 4, cudaMemcpyDeviceToDevice, s));
    verify_forward(R);
    ik::accept_rows(vtok_, vy_, R, vst_, d_gen_, nacc_hist_, s);
    const int* nc = vst_ + 2;
    ik::conv_commit_n(khist_, RR, cst_k_, 1024, 48, nc, s);
    ik::conv_commit_n(vhist_, RR, cst_v_, 1024, 48, nc, s);
    ik::conv_commit_n(hist_a_, RR, cst_a_, ik::H, 48, nc, s);
    ik::conv_commit_n(hist_m_, RR, cst_m_, ik::H, 48, nc, s);
    ik::ring_commit(lk_, lv_, ltk_, ltv_, RR, n_lslots_, RR, vst_ + 1, nc, s);
    mtp_ingest(featv_, R, 0, vst_ + 1, nc, nc);
    ik::advance_pos(vst_, s);
}
cudaGraphExec_t mtp_capture(int R) {
    QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    mtp_cycle(R);
    cudaGraph_t g;
    QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
    cudaGraphExec_t ge;
    QV_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    QV_CUDA_CHECK(cudaGraphDestroy(g));
    return ge;
}


// ---- lossless speculative sampling at T > 0 (include/qv/spec_sampling.h). The drafter is deterministic (greedy chain), so its proposal distribution is a point
// mass q = delta(d): a draft is accepted with probability p(d) and on rejection the token is drawn from p with d removed (exactly the residual max(0, p - q) / norm).
// Cycle = graph A (draft + verify + candidate extraction, results copied to pinned host memory) -> one host sync -> host accept/reject -> graph B (commit n+1 rows,
// ingest) -> next graph A, queued before the host waits again (one sync per cycle).
void sp_extract_async(const float* dlog, int rows, float margin) {
    cudaStream_t s = stream_;
    const int N = cfg_.ink.unpadded_vocab;
    extract_top_candidates_batch(dlog, N, rows, margin, SP_MAXC, d_scid_, d_scval_, d_smeta_, s);
    QV_CUDA_CHECK(cudaMemcpyAsync(h_smeta_, d_smeta_, (size_t)rows * sizeof(ik_CandMeta), cudaMemcpyDeviceToHost, s));
    QV_CUDA_CHECK(cudaMemcpy2DAsync(h_scid_, SP_MAXC * 4, d_scid_, SP_MAXC * 4, SP_FAST * 4, rows, cudaMemcpyDeviceToHost, s));
    QV_CUDA_CHECK(cudaMemcpy2DAsync(h_scval_, SP_MAXC * 4, d_scval_, SP_MAXC * 4, SP_FAST * 4, rows, cudaMemcpyDeviceToHost, s));
}
void sp_tail_sync(int rows) {  // after the stream sync: fetch the candidates beyond the first SP_FAST of rows that have more
    bool any = false;
    for (int t = 0; t < rows; ++t) {
        const int cnt = h_smeta_[t].count;
        if (cnt > SP_FAST && cnt <= SP_MAXC) {
            QV_CUDA_CHECK(cudaMemcpyAsync(h_scid_ + (size_t)t * SP_MAXC + SP_FAST, d_scid_ + (size_t)t * SP_MAXC + SP_FAST, (size_t)(cnt - SP_FAST) * 4, cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_scval_ + (size_t)t * SP_MAXC + SP_FAST, d_scval_ + (size_t)t * SP_MAXC + SP_FAST, (size_t)(cnt - SP_FAST) * 4, cudaMemcpyDeviceToHost, stream_));
            any = true;
        }
    }
    if (any) QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}
void sp_build_row(const float* dlog, int row, float margin, const SpecSamplingParams& sp, SpecDist& out, TokenConstraint* tc = nullptr) {
    const int N = cfg_.ink.unpadded_vocab;
    bool ok = false;
    if (margin >= 0.0f) {
        const ik_CandMeta& md = h_smeta_[row];
        ok = md.count > 0 && md.count <= SP_MAXC &&
             build_dist_from_candidates(h_scid_ + (size_t)row * SP_MAXC, h_scval_ + (size_t)row * SP_MAXC, md.count, md.mx, (double)md.sum_exp, sp, out, tc, margin);
    }
    if (!ok) {
        if ((int)h_sfull_.size() < N) h_sfull_.resize(N);
        QV_CUDA_CHECK(cudaMemcpyAsync(h_sfull_.data(), dlog + (size_t)row * N, (size_t)N * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        build_dist_dense(h_sfull_.data(), N, sp, out, tc);
    }
}
void sp_cycleA(int R, float margin, bool sampled) {
    cudaStream_t s = stream_;
    dsp_on_ = sampled;
    draft_block(R - 1);
    dsp_on_ = false;
    QV_CUDA_CHECK(cudaMemcpyAsync(vtok_, vst_, 4, cudaMemcpyDeviceToDevice, s));
    verify_forward(R);
    if (margin >= 0.0f) sp_extract_async(logitsv_, R, margin);
    QV_CUDA_CHECK(cudaMemcpyAsync(h_svtok_, vtok_, R * sizeof(int), cudaMemcpyDeviceToHost, s));
    if (sampled) {
        QV_CUDA_CHECK(cudaMemcpyAsync(h_dq_cnt_, d_dq_cnt_, (R - 1) * sizeof(int), cudaMemcpyDeviceToHost, s));
        QV_CUDA_CHECK(cudaMemcpyAsync(h_dq_id_, d_dq_id_, (size_t)(R - 1) * 256 * sizeof(int), cudaMemcpyDeviceToHost, s));
        QV_CUDA_CHECK(cudaMemcpyAsync(h_dq_prob_, d_dq_prob_, (size_t)(R - 1) * 256 * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
}
void sp_cycleB(int R) {
    cudaStream_t s = stream_;
    const int* nc = vst_ + 2;
    ik::conv_commit_n(khist_, RR, cst_k_, 1024, 48, nc, s);
    ik::conv_commit_n(vhist_, RR, cst_v_, 1024, 48, nc, s);
    ik::conv_commit_n(hist_a_, RR, cst_a_, ik::H, 48, nc, s);
    ik::conv_commit_n(hist_m_, RR, cst_m_, ik::H, 48, nc, s);
    ik::ring_commit(lk_, lv_, ltk_, ltv_, RR, n_lslots_, RR, vst_ + 1, nc, s);
    mtp_ingest(featv_, R, 0, vst_ + 1, nc, nc);
    ik::advance_pos(vst_, s);
}
template <class F>
cudaGraphExec_t sp_capture(F fn) {
    QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    fn();
    cudaGraph_t g;
    QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
    cudaGraphExec_t ge;
    QV_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    QV_CUDA_CHECK(cudaGraphDestroy(g));
    return ge;
}
void sp_graphs(int R, float margin, bool sampled, const ik::DraftSampleParams& dp) {
    if (!g_spB_[R]) g_spB_[R] = sp_capture([&] { sp_cycleB(R); });
    if (!g_spA_[R] || spA_margin_[R] != margin || spA_sampled_[R] != sampled || (sampled && !(spA_dsp_[R] == dp))) {
        if (g_spA_[R]) cudaGraphExecDestroy(g_spA_[R]);
        dsp_ = dp;
        g_spA_[R] = sp_capture([&] { sp_cycleA(R, margin, sampled); });
        spA_margin_[R] = margin; spA_sampled_[R] = sampled; spA_dsp_[R] = dp;
    }
}
// verify width for `rem` tokens still to produce (a cycle emits at most R tokens, never more than rem): 0 = use a plain step
static int sp_pick_r(int rem, int rmax) {
    const int r = std::min(rem, rmax);
    return r >= 8 ? 8 : r >= 4 ? 4 : r >= 2 ? r : 0;
}
// Adaptive width (p29 S3): per-position conditional acceptance EMA, cost model from the QV_MTP_PROF numbers at short context (ms): plain sampled step, draft block, verify R.
float sp_ema_[8] = {0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f};
int sp_plain_run_ = 0;
int sp_policy(int rem, int rmax) {
    static const int adapt = std::getenv("QV_SPEC_ADAPT") ? std::atoi(std::getenv("QV_SPEC_ADAPT")) : 1;
    if (!adapt) return sp_pick_r(rem, rmax);
    const int cap = sp_pick_r(rem, rmax);
    if (cap < 2) return 0;
    if (sp_plain_run_ >= 12) return cap > 4 ? 4 : cap;  // exploration: refresh the acceptance estimate every 12 plain tokens
    static const float verify_ms[9] = {0, 0, 12.6f, 14.5f, 17.1f, 0, 0, 0, 26.7f};
    float best = 1.0f / 7.7f;  // tokens per ms of a plain sampled step
    int best_r = 0;
    for (int R : {2, 3, 4, 8}) {
        if (R > cap) continue;
        float e = 1.0f, pr = 1.0f;
        for (int j = 0; j < R - 1; ++j) { pr *= sp_ema_[j]; e += pr; }
        const float cost = 2.2f + 0.17f * (R - 1) + verify_ms[R] + 0.35f;
        if (e / cost > best) { best = e / cost; best_r = R; }
    }
    return best_r;
}
void sp_update_ema(int R, int n_acc) {
    const int tested = std::min(n_acc + 1, R - 1);
    for (int j = 0; j < tested; ++j) sp_ema_[j] = 0.85f * sp_ema_[j] + 0.15f * (j < n_acc ? 1.0f : 0.0f);
}
void mtp_decode_spec(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out, TokenConstraint* tc = nullptr) {
    const int Rmax = std::min(8, std::max(2, opt_.mtp_depth + 1));
    const int Rtop = Rmax <= 2 ? 2 : Rmax == 3 ? 3 : Rmax <= 4 ? 4 : 8;
    const float margin = spec_candidate_margin(sp);
    // QV_SPEC_DRAFT: point (default: greedy drafts, point-mass proposal) | sample (drafts drawn from the drafter distribution; measured the same acceptance, see docs/INKLING_MTP.md)
    const char* dm = std::getenv("QV_SPEC_DRAFT");
    const bool sampled = sp.temperature > 0.0f && dm && std::string(dm) == "sample";
    ik::DraftSampleParams dp;
    dp.temperature = sp.temperature; dp.top_k = sp.top_k; dp.top_p = sp.top_p; dp.min_p = sp.min_p; dp.min_keep = sp.min_keep; dp.max_extract = SP_MAXC;
    dp.margin = (sp.min_p > 0.0f && sp.min_keep <= 1) ? std::max(3.5f, -std::log(sp.min_p) + 0.1f) : std::min(9.0f, std::max(4.0f, 7.5f * sp.temperature));
    if (start_pos + n + RR > opt_.max_ctx) throw std::runtime_error("context overflow");
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::mt19937_64 rng(sp.seed);
    out.clear();
    out.reserve(n);
    int cur = first, p = start_pos;
    bool a_inflight = false, st_valid = false;
    int R_inflight = 0;
    auto t0 = clk::now();
    while ((int)out.size() < n) {
        const int R = a_inflight ? R_inflight : sp_policy(n - (int)out.size(), Rtop);
        if (R == 0) {
            // plain step sampled on the host from GPU-extracted candidates (state equals plain decode: all but the last output processed)
            const auto tp0 = clk::now();
            mtp_plain_step(cur, p);
            if (margin >= 0.0f) {
                sp_extract_async(logits_, 1, margin);
                QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
                sp_tail_sync(1);
            }
            SpecDist pj;
            sp_build_row(logits_, 0, margin, sp, pj, tc);
            cur = sample_dist(pj, rng);
            out.push_back(cur);
            ++p;
            ++sp_plain_run_;
            st_valid = false;
            mstats_.plain_steps += 1;
            mstats_.plain_ms += ms(tp0, clk::now());
            if (tc) { (void)tc->try_accept(cur); if (tc->is_eog(cur) || tc->is_dead()) return; }
            continue;
        }
        sp_plain_run_ = 0;
        sp_graphs(R, margin, sampled, dp);
        if (!st_valid) {
            h_sst_[0] = cur; h_sst_[1] = p; h_sst_[2] = 0;
            { const unsigned long long sd = rng(); std::memcpy(h_sst_ + 6, &sd, 8); }
            QV_CUDA_CHECK(cudaMemcpyAsync(vst_, h_sst_, 32, cudaMemcpyHostToDevice, stream_));
            QV_CUDA_CHECK(cudaStreamSynchronize(stream_));  // h_sst_ is reused below
            st_valid = true;
            a_inflight = false;
        }
        if (!a_inflight) { t0 = clk::now(); QV_CUDA_CHECK(cudaGraphLaunch(g_spA_[R], stream_)); }
        a_inflight = false;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        sp_tail_sync(R);
        int drafts[8];
        for (int j = 0; j < R - 1; ++j) drafts[j] = h_svtok_[j + 1];
        int n_acc = 0;
        int seq[8];
        int m;
        bool stop = false;
        if (tc) {  // constrained decoding (grammar): sequential rounds, the constraint advances with every accepted token
            m = 0;
            for (int j = 0; j < R; ++j) {
                SpecDist pj;
                sp_build_row(logitsv_, j, margin, sp, pj, tc);
                int tok_j;
                bool acc = false;
                if (j < R - 1) {
                    SpecDist q;
                    if (!sampled) {
                        q.sp.assign(1, {drafts[j], 1.0f});
                    } else {
                        const int qc = std::clamp(h_dq_cnt_[j], 0, 256);
                        q.sp.resize((size_t)qc);
                        for (int i = 0; i < qc; ++i) q.sp[(size_t)i] = {h_dq_id_[j * 256 + i], h_dq_prob_[j * 256 + i]};
                        std::sort(q.sp.begin(), q.sp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                    }
                    acc = spec_accept(pj, q, drafts[j], rng);
                    tok_j = acc ? drafts[j] : sample_residual(pj, q, rng);
                } else {
                    tok_j = sample_dist(pj, rng);
                }
                seq[j] = tok_j;
                (void)tc->try_accept(tok_j);
                if (acc) n_acc = j + 1;
                m = j;
                if (tc->is_eog(tok_j) || tc->is_dead()) { stop = true; break; }
                if (!acc) break;
            }
        } else if (!sampled) {
            m = spec_round_point(R - 1, drafts, [&](int j, SpecDist& d) { sp_build_row(logitsv_, j, margin, sp, d); }, rng, seq, &n_acc);
        } else {
            m = spec_round_q(R - 1, drafts, [&](int j, SpecDist& d) { sp_build_row(logitsv_, j, margin, sp, d); },
                             [&](int j, SpecDist& q) {
                                 const int qc = std::clamp(h_dq_cnt_[j], 0, 256);
                                 q.dense = false; q.dn.clear(); q.sp.resize((size_t)qc);
                                 for (int i = 0; i < qc; ++i) q.sp[(size_t)i] = {h_dq_id_[j * 256 + i], h_dq_prob_[j * 256 + i]};
                                 std::sort(q.sp.begin(), q.sp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                             }, rng, seq, &n_acc);
        }
        const int tok_final = seq[m];
        for (int j = 0; j <= m; ++j) out.push_back(seq[j]);
        h_sst_[0] = tok_final; h_sst_[1] = p; h_sst_[2] = m + 1;
        { const unsigned long long sd = rng(); std::memcpy(h_sst_ + 6, &sd, 8); }
        QV_CUDA_CHECK(cudaMemcpyAsync(vst_, h_sst_, 32, cudaMemcpyHostToDevice, stream_));
        QV_CUDA_CHECK(cudaGraphLaunch(g_spB_[R], stream_));
        cur = tok_final;
        p += m + 1;
        mstats_.cycles += 1; mstats_.drafted += R - 1; mstats_.accepted += n_acc;
        mstats_.verify_ms += ms(t0, clk::now());
        sp_update_ema(R, n_acc);
        if (stop) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); return; }
        // queue the next cycle before the host waits again when it will be a speculative one of the same state chain
        const int Rn = sp_policy(n - (int)out.size(), Rtop);
        if (Rn) {
            sp_graphs(Rn, margin, sampled, dp);
            t0 = clk::now();
            QV_CUDA_CHECK(cudaGraphLaunch(g_spA_[Rn], stream_));
            a_inflight = true;
            R_inflight = Rn;
        } else {
            st_valid = false;
        }
    }
    QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// plain steps for the tail (and for tests): same state as the speculative path (features -> drafter context)
void mtp_plain_step(int token, int pos) {
    int tp[2] = {token, pos};
    QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, tp, 8, cudaMemcpyHostToDevice, stream_));
    QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    forward();
    mtp_ingest(featv_, 1, 0, d_pos_, nullptr, nullptr);
}

// requested tokens: out[i] = i-th successor of `first` (token at start_pos, not yet in the cache). Exactly n tokens; the state afterwards equals
// plain decode_greedy's (all but the last output processed).
void mtp_decode(int first, int start_pos, int n, std::vector<int>& out) {
    const int Rmax = std::min(8, std::max(2, opt_.mtp_depth + 1));
    const int Rv = Rmax <= 2 ? 2 : Rmax == 3 ? 3 : Rmax <= 4 ? 4 : 8;
    int st[8] = {first, start_pos, 0, 0, 0, 0, 0, 0};
    if (std::getenv("QV_MTP_PROF")) { mtp_profile(); std::exit(0); }
    if (start_pos + n + RR > opt_.max_ctx) throw std::runtime_error("context overflow");
    QV_CUDA_CHECK(cudaMemcpyAsync(vst_, st, sizeof st, cudaMemcpyHostToDevice, stream_));
    if (!g_mtp_[Rv]) g_mtp_[Rv] = mtp_capture(Rv);
    int done = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (n - done >= Rv) {
        QV_CUDA_CHECK(cudaGraphLaunch(g_mtp_[Rv], stream_));
        QV_CUDA_CHECK(cudaMemcpyAsync(st, vst_, sizeof st, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        done = st[3];
        mstats_.cycles += 1; mstats_.drafted += Rv - 1;
    }
    mstats_.verify_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    out.resize(n);
    if (done > 0) QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_gen_, done * 4, cudaMemcpyDeviceToHost, stream_));
    QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    mstats_.accepted += st[5];
    // tail: plain steps from the current anchor
    int tok = st[0], pos = st[1];
    t0 = std::chrono::steady_clock::now();
    if (done == 0) { tok = first; pos = start_pos; }
    while (done < n) {
        mtp_plain_step(tok, pos);
        argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_arg_, nullptr, nullptr, stream_);
        int r;
        QV_CUDA_CHECK(cudaMemcpyAsync(&r, d_arg_, 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        out[done++] = r;
        tok = r;
        ++pos;
        mstats_.plain_steps += 1;
    }
    mstats_.plain_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// QV_MTP_PROF=1: per-stage GPU time (each stage in its own CUDA graph, 50 replays; state contents irrelevant, nothing is committed)
void mtp_profile() {
    cudaStream_t s = stream_;
    auto cap = [&](auto fn) {
        QV_CUDA_CHECK(cudaStreamSynchronize(s));
        QV_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        fn();
        cudaGraph_t g; cudaGraphExec_t ge;
        QV_CUDA_CHECK(cudaStreamEndCapture(s, &g));
        QV_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        QV_CUDA_CHECK(cudaGraphDestroy(g));
        return ge;
    };
    auto time = [&](cudaGraphExec_t ge) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s);
        cudaEventRecord(a, s);
        for (int i = 0; i < 50; ++i) cudaGraphLaunch(ge, s);
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b); cudaGraphExecDestroy(ge);
        return ms / 50;
    };
    int st[8] = {1000, 0, 0, 0, 0, 0, 0, 0};
    st[1] = 600;  // any position inside the prefilled context
    QV_CUDA_CHECK(cudaMemcpyAsync(vst_, st, sizeof st, cudaMemcpyHostToDevice, s));
    if (!std::getenv("QV_MTP_PROF_R")) std::fprintf(stderr, "[mtp-prof] draft_block %.3f ms\n", time(cap([&] { draft_block(); })));
    const char* only = std::getenv("QV_MTP_PROF_R");  // optional: profile one width only (for nsys)
    for (int R : {1, 2, 3, 4, 8}) {
        if (only && std::atoi(only) != R) continue;
        if (R == 1) { std::fprintf(stderr, "[mtp-prof] plain 1-token step %.3f ms\n", time(cap([&] { forward(); }))); continue; }
        std::fprintf(stderr, "[mtp-prof] verify R=%d %.3f ms\n", R, time(cap([&] { verify_forward(R); })));
    }
    if (!only) std::fprintf(stderr, "[mtp-prof] accept+commit+ingest(R=8) %.3f ms\n", time(cap([&] {
        ik::accept_rows(vtok_, vy_, 8, vst_, d_gen_, nacc_hist_, s);
        const int* nc = vst_ + 2;
        ik::conv_commit_n(khist_, RR, cst_k_, 1024, 48, nc, s); ik::conv_commit_n(vhist_, RR, cst_v_, 1024, 48, nc, s);
        ik::conv_commit_n(hist_a_, RR, cst_a_, ik::H, 48, nc, s); ik::conv_commit_n(hist_m_, RR, cst_m_, ik::H, 48, nc, s);
        ik::ring_commit(lk_, lv_, ltk_, ltv_, RR, n_lslots_, RR, vst_ + 1, nc, s);
        mtp_ingest(featv_, 8, 0, vst_ + 1, nc, nc);
    })));
}
