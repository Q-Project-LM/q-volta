// Lossless speculative sampling helpers shared across backends (CUDA, Mock, tests).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace qv {

struct SpecSamplingParams {
    float temperature = 0.0f;  // <= 0 means greedy
    int top_k = 0;             // <= 0 disabled
    float top_p = 1.0f;        // >= 1.0 disabled
    float min_p = 0.0f;        // <= 0.0 disabled
    int min_keep = 0;
    uint64_t seed = 0;
};

struct TokenConstraint {
    virtual ~TokenConstraint() = default;
    virtual bool awaiting() const = 0;
    virtual bool is_dead() const = 0;
    virtual bool is_eog(int tok) const = 0;
    virtual bool probe_allowed(int tok) const = 0;
    virtual bool try_accept(int tok) = 0;
    virtual void mask_logits(float* logits, size_t n) = 0;
};

struct SpecDist {
    bool dense = false;
    std::vector<std::pair<int, float>> sp;  // sorted ascending by token id; probs sum to ~1
    std::vector<float> dn;                  // dense probs of size V when dense == true

    float prob_of(int id) const {
        if (id < 0) return 0.0f;
        if (dense) return (size_t)id < dn.size() ? dn[(size_t)id] : 0.0f;
        auto it = std::lower_bound(sp.begin(), sp.end(), id,
                                   [](const std::pair<int, float>& a, int key) { return a.first < key; });
        return (it != sp.end() && it->first == id) ? it->second : 0.0f;
    }
};

// Margin (mx - val <= margin) for GPU candidate extraction; < 0 means go straight to dense softmax.
inline float spec_candidate_margin(const SpecSamplingParams& p) {
    if (p.temperature <= 0.0f) return 0.0f;
    if (p.top_k <= 0 && p.top_p >= 1.0f && p.min_p <= 0.0f) {
        return (p.temperature <= 1.5f) ? (24.0f * p.temperature) : -1.0f;
    }
    if (p.min_p > 0.0f && p.top_p >= 1.0f && p.top_k <= 0 && p.min_keep <= 1) {
        return -std::log(p.min_p) + 1e-3f;
    }
    const float minp_margin = p.min_p > 0.0f ? (-std::log(p.min_p) + 1e-3f) : 10.0f;
    if (p.top_k > 0) return std::max(14.0f, minp_margin);
    return std::max(10.0f, minp_margin);
}

namespace detail {
struct SpecCand {
    int id;
    float val;
};

inline void apply_filters_and_temp(std::vector<SpecCand>& cur, double z_full, float mx, bool full_vocab,
                                   const SpecSamplingParams& p, SpecDist& out, bool& need_fallback) {
    need_fallback = false;
    std::sort(cur.begin(), cur.end(), [](const SpecCand& a, const SpecCand& b) {
        return a.val > b.val || (a.val == b.val && a.id < b.id);
    });
    const size_t min_k = (size_t)std::max(1, p.min_keep);
    bool trimmed = false;
    if (p.top_k > 0) {
        const size_t k = std::max<size_t>((size_t)p.top_k, min_k);
        if (!full_vocab && cur.size() < k) { need_fallback = true; return; }
        if (k < cur.size()) { cur.resize(k); trimmed = true; }
    }
    if (p.top_p < 1.0f) {
        double denom = 0.0;
        if (trimmed || (full_vocab && p.top_k > 0)) {
            const float cmax = cur[0].val;
            for (const auto& c : cur) denom += std::exp((double)c.val - (double)cmax);
            double cum = 0.0;
            size_t keep = cur.size();
            for (size_t i = 0; i < cur.size(); ++i) {
                cum += std::exp((double)cur[i].val - (double)cmax) / denom;
                if (cum >= (double)p.top_p && i + 1 >= min_k) { keep = i + 1; break; }
            }
            cur.resize(keep);
        } else {
            denom = z_full;
            if (!(denom > 0.0) || !std::isfinite(denom)) { need_fallback = !full_vocab; return; }
            double cum = 0.0;
            size_t keep = cur.size();
            bool reached = false;
            for (size_t i = 0; i < cur.size(); ++i) {
                cum += std::exp((double)cur[i].val - (double)mx) / denom;
                if (cum >= (double)p.top_p && i + 1 >= min_k) { keep = i + 1; reached = true; break; }
            }
            if (!full_vocab && !reached) { need_fallback = true; return; }
            cur.resize(keep);
        }
    }
    if (p.min_p > 0.0f && !cur.empty()) {
        const float cmax = cur[0].val;
        size_t keep = 0;
        while (keep < cur.size() && std::exp((double)cur[keep].val - (double)cmax) >= (double)p.min_p) ++keep;
        keep = std::max(keep, std::min(cur.size(), min_k));
        if (!full_vocab && keep < min_k && cur.size() < min_k) { need_fallback = true; return; }
        cur.resize(keep);
    }
    if (cur.empty()) { need_fallback = !full_vocab; return; }
    out.dense = false;
    out.dn.clear();
    if (p.temperature <= 0.0f || !std::isfinite(cur[0].val)) {
        out.sp.assign(1, {cur[0].id, 1.0f});
        return;
    }
    const float cmax = cur[0].val;
    double sum_w = 0.0;
    std::vector<double> w(cur.size());
    const double inv_t = 1.0 / (double)p.temperature;
    for (size_t i = 0; i < cur.size(); ++i) {
        w[i] = std::exp(((double)cur[i].val - (double)cmax) * inv_t);
        sum_w += w[i];
    }
    out.sp.resize(cur.size());
    const double inv_sum = sum_w > 0.0 ? (1.0 / sum_w) : 0.0;
    for (size_t i = 0; i < cur.size(); ++i) out.sp[i] = {cur[i].id, (float)(w[i] * inv_sum)};
    std::sort(out.sp.begin(), out.sp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
}
}  // namespace detail

// Build SpecDist from GPU-extracted candidates (all tokens with val >= mx - margin). Returns false when a full-vocab fallback is required.
inline bool build_dist_from_candidates(const int* ids, const float* vals, int count, float mx, double z_full,
                                       const SpecSamplingParams& p, SpecDist& out, TokenConstraint* tc = nullptr,
                                       float margin = -1.0f) {
    if (count <= 0 || !std::isfinite(mx)) return false;
    const size_t min_k = (size_t)std::max(1, p.min_keep);
    const bool active_tc = tc && !tc->awaiting();
    if (active_tc && tc->is_dead()) return false;
    std::vector<detail::SpecCand> cur;
    cur.reserve((size_t)count);
    bool any_rejected = false;
    if (active_tc) {
        for (int i = 0; i < count; ++i) {
            if (tc->probe_allowed(ids[i])) cur.push_back({ids[i], vals[i]});
            else any_rejected = true;
        }
    } else {
        for (int i = 0; i < count; ++i) cur.push_back({ids[i], vals[i]});
    }
    if (cur.empty()) return false;
    if (any_rejected && p.top_k <= 0 && p.top_p < 1.0f) return false;  // z_full includes disallowed tokens; use dense masked fallback
    if (p.temperature <= 0.0f) {
        int best_id = cur[0].id;
        float best_val = cur[0].val;
        for (size_t i = 1; i < cur.size(); ++i) {
            if (cur[i].val > best_val || (cur[i].val == best_val && cur[i].id < best_id)) {
                best_val = cur[i].val;
                best_id = cur[i].id;
            }
        }
        out.dense = false;
        out.dn.clear();
        out.sp.assign(1, {best_id, 1.0f});
        return true;
    }
    if (min_k > 1 && cur.size() < min_k) return false;
    float cand_mx = mx;
    if (any_rejected) {
        cand_mx = cur[0].val;
        for (size_t i = 1; i < cur.size(); ++i) cand_mx = std::max(cand_mx, cur[i].val);
        const bool pure_temp = (p.top_k <= 0 && p.top_p >= 1.0f && p.min_p <= 0.0f);
        if (pure_temp && cand_mx < mx) return false;
    }
    if (p.top_k > 0) {
        const size_t k = std::max<size_t>((size_t)p.top_k, min_k);
        if (cur.size() < k) {
            if (margin <= 0.0f) return false;
            const double eff_margin = (double)margin - ((double)mx - (double)cand_mx);
            if (eff_margin <= 0.0) return false;
            const double tail_one = std::exp(-eff_margin);
            const double eps = (double)(k - cur.size()) * tail_one;
            std::sort(cur.begin(), cur.end(), [](const detail::SpecCand& a, const detail::SpecCand& b) {
                return a.val > b.val || (a.val == b.val && a.id < b.id);
            });
            if (p.top_p < 1.0f) {
                double sum_s = 0.0;
                for (const auto& c : cur) sum_s += std::exp((double)c.val - (double)cand_mx);
                if (!(sum_s > 0.0) || !std::isfinite(sum_s)) return false;
                double cum = 0.0;
                double prev_cum = 0.0;
                size_t keep = cur.size();
                bool found = false;
                for (size_t i = 0; i < cur.size(); ++i) {
                    const double w = std::exp((double)cur[i].val - (double)cand_mx);
                    if (i + 1 >= min_k && !found) {
                        const double next_cum = cum + w;
                        if (next_cum / (sum_s + eps) >= (double)p.top_p) {
                            keep = i + 1;
                            prev_cum = cum;
                            found = true;
                            break;
                        }
                    }
                    cum += w;
                }
                if (!found || (keep > min_k && prev_cum / sum_s >= (double)p.top_p)) return false;
                cur.resize(keep);
            } else if (p.min_p > 0.0f) {
                if (tail_one >= (double)p.min_p) return false;
            } else {
                return false;
            }
            if (p.min_p > 0.0f && !cur.empty()) {
                size_t keep = 0;
                while (keep < cur.size() && std::exp((double)cur[keep].val - (double)cand_mx) >= (double)p.min_p) ++keep;
                keep = std::max(keep, std::min(cur.size(), min_k));
                cur.resize(keep);
            }
            if (cur.empty()) return false;
            double sum_w = 0.0;
            std::vector<double> w(cur.size());
            const double inv_t = 1.0 / (double)p.temperature;
            for (size_t i = 0; i < cur.size(); ++i) {
                w[i] = std::exp(((double)cur[i].val - (double)cand_mx) * inv_t);
                sum_w += w[i];
            }
            out.dense = false;
            out.dn.clear();
            out.sp.resize(cur.size());
            const double inv_sum = sum_w > 0.0 ? (1.0 / sum_w) : 0.0;
            for (size_t i = 0; i < cur.size(); ++i) out.sp[i] = {cur[i].id, (float)(w[i] * inv_sum)};
            std::sort(out.sp.begin(), out.sp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            return true;
        }
    }
    bool need_fallback = false;
    detail::apply_filters_and_temp(cur, z_full, cand_mx, false, p, out, need_fallback);
    return !need_fallback;
}

// Build exact SpecDist over full logits array of size V.
inline void build_dist_dense(const float* logits, int V, const SpecSamplingParams& p, SpecDist& out, TokenConstraint* tc = nullptr) {
    if (V <= 0) {
        out.dense = false;
        out.dn.clear();
        out.sp.clear();
        return;
    }
    std::vector<float> masked;
    if (tc && !tc->awaiting()) {
        masked.assign(logits, logits + V);
        tc->mask_logits(masked.data(), (size_t)V);
        logits = masked.data();
    }
    if (p.temperature <= 0.0f) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < V; ++i) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
        }
        out.dense = false;
        out.dn.clear();
        out.sp.assign(1, {best_id, 1.0f});
        return;
    }
    if (p.top_k <= 0 && p.top_p >= 1.0f && p.min_p <= 0.0f) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < V; ++i) mx = std::max(mx, logits[i]);
        if (!std::isfinite(mx)) {
            out.dense = false;
            out.dn.clear();
            out.sp.assign(1, {0, 1.0f});
            return;
        }
        const double inv_t = 1.0 / (double)p.temperature;
        if (V > 256) {
            out.sp.clear();
            out.sp.reserve(256);
            double sum_sp = 0.0;
            bool overflow = false;
            for (int i = 0; i < V; ++i) {
                if (!std::isfinite(logits[i])) continue;
                const double z = ((double)logits[i] - (double)mx) * inv_t;
                if (z > -36.0) {
                    if (out.sp.size() >= 2048) { overflow = true; break; }
                    const float w = (float)std::exp(z);
                    out.sp.emplace_back(i, w);
                    sum_sp += w;
                }
            }
            if (!overflow && !out.sp.empty() && sum_sp > 0.0) {
                const double inv_s = 1.0 / sum_sp;
                for (auto& kv : out.sp) kv.second = (float)(kv.second * inv_s);
                out.dense = false;
                out.dn.clear();
                return;
            }
            out.sp.clear();
        }
        out.dense = true;
        out.sp.clear();
        out.dn.resize((size_t)V);
        double sum = 0.0;
        for (int i = 0; i < V; ++i) {
            float w = 0.0f;
            if (std::isfinite(logits[i])) {
                const double z = ((double)logits[i] - (double)mx) * inv_t;
                if (z > -36.0) w = (float)std::exp(z);
            }
            out.dn[(size_t)i] = w;
            sum += w;
        }
        if (sum > 0.0) {
            const double inv_s = 1.0 / sum;
            for (float& v : out.dn) v = (float)(v * inv_s);
        } else {
            out.dn[0] = 1.0f;
        }
        return;
    }
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < V; ++i) mx = std::max(mx, logits[i]);
    const size_t min_k = (size_t)std::max(1, p.min_keep);
    auto gt = [](const detail::SpecCand& a, const detail::SpecCand& b) {
        return a.val > b.val || (a.val == b.val && a.id < b.id);
    };
    if (!std::isfinite(mx)) {
        out.dense = false;
        out.dn.clear();
        out.sp.assign(1, {0, 1.0f});
        return;
    }
    auto fill_finite_or_all = [&](std::vector<detail::SpecCand>& dst, size_t req_k) {
        dst.clear();
        dst.reserve(std::min((size_t)V, std::max<size_t>(req_k, 1024)));
        for (int i = 0; i < V; ++i) {
            if (std::isfinite(logits[i])) dst.push_back({i, logits[i]});
        }
        if (dst.size() < req_k) {
            dst.resize((size_t)V);
            for (int i = 0; i < V; ++i) dst[(size_t)i] = {i, logits[i]};
        }
    };
    std::vector<detail::SpecCand> cur;
    if (p.top_k > 0) {
        const size_t k = std::max<size_t>((size_t)p.top_k, min_k);
        fill_finite_or_all(cur, k);
        if (k < cur.size()) {
            std::nth_element(cur.begin(), cur.begin() + k, cur.end(), gt);
            cur.resize(k);
        }
        bool need_fallback = false;
        detail::apply_filters_and_temp(cur, 0.0, mx, true, p, out, need_fallback);
    } else if (p.top_p < 1.0f) {
        double z_full = 0.0;
        cur.reserve(256);
        for (int i = 0; i < V; ++i) {
            if (!std::isfinite(logits[i])) continue;
            const float d = logits[i] - mx;
            if (d > -36.0f) z_full += std::exp((double)d);
            if (d >= -14.0f) cur.push_back({i, logits[i]});
        }
        bool need_fallback = false;
        if (!cur.empty()) {
            detail::apply_filters_and_temp(cur, z_full, mx, false, p, out, need_fallback);
        } else {
            need_fallback = true;
        }
        if (need_fallback) {
            fill_finite_or_all(cur, min_k);
            detail::apply_filters_and_temp(cur, z_full, mx, true, p, out, need_fallback);
        }
    } else {
        const float thr = mx + std::log(p.min_p) - 1e-4f;
        cur.reserve(256);
        for (int i = 0; i < V; ++i) {
            if (std::isfinite(logits[i]) && logits[i] >= thr) cur.push_back({i, logits[i]});
        }
        bool need_fallback = false;
        if (cur.size() >= min_k) {
            detail::apply_filters_and_temp(cur, 0.0, mx, true, p, out, need_fallback);
        } else {
            fill_finite_or_all(cur, min_k);
            if (min_k < cur.size()) {
                std::nth_element(cur.begin(), cur.begin() + min_k, cur.end(), gt);
                cur.resize(min_k);
            }
            detail::apply_filters_and_temp(cur, 0.0, mx, true, p, out, need_fallback);
        }
    }
}

inline double spec_uniform(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

inline int sample_dist(const SpecDist& d, std::mt19937_64& rng) {
    const double u = spec_uniform(rng);
    double acc = 0.0;
    if (d.dense) {
        for (size_t i = 0; i < d.dn.size(); ++i) {
            acc += d.dn[i];
            if (acc >= u) return (int)i;
        }
        return d.dn.empty() ? 0 : (int)d.dn.size() - 1;
    }
    for (size_t i = 0; i < d.sp.size(); ++i) {
        acc += d.sp[i].second;
        if (acc >= u) return d.sp[i].first;
    }
    return d.sp.empty() ? 0 : d.sp.back().first;
}

inline bool spec_accept(const SpecDist& p, const SpecDist& q, int tok, std::mt19937_64& rng) {
    const double px = p.prob_of(tok);
    const double qx = q.prob_of(tok);
    if (qx <= 0.0) return false;
    if (px >= qx) return true;
    return spec_uniform(rng) * qx < px;
}

inline int sample_residual(const SpecDist& p, const SpecDist& q, std::mt19937_64& rng) {
    if (p.dense) {
        const size_t V = p.dn.size();
        double sum = 0.0;
        for (size_t i = 0; i < V; ++i) {
            const double d = (double)p.dn[i] - (double)q.prob_of((int)i);
            if (d > 0.0) sum += d;
        }
        if (sum <= 1e-12) return sample_dist(p, rng);
        const double tgt = spec_uniform(rng) * sum;
        double acc = 0.0;
        int last = 0;
        for (size_t i = 0; i < V; ++i) {
            const double d = (double)p.dn[i] - (double)q.prob_of((int)i);
            if (d > 0.0) {
                last = (int)i;
                acc += d;
                if (acc >= tgt) return (int)i;
            }
        }
        return last;
    }
    double sum = 0.0;
    std::vector<std::pair<int, double>> diffs;
    diffs.reserve(p.sp.size());
    if (!q.dense) {
        size_t qi = 0;
        for (const auto& kv : p.sp) {
            while (qi < q.sp.size() && q.sp[qi].first < kv.first) ++qi;
            const double qv = (qi < q.sp.size() && q.sp[qi].first == kv.first) ? (double)q.sp[qi].second : 0.0;
            const double d = (double)kv.second - qv;
            if (d > 0.0) { diffs.emplace_back(kv.first, d); sum += d; }
        }
    } else {
        for (const auto& kv : p.sp) {
            const double d = (double)kv.second - (double)q.prob_of(kv.first);
            if (d > 0.0) { diffs.emplace_back(kv.first, d); sum += d; }
        }
    }
    if (sum <= 1e-12 || diffs.empty()) return sample_dist(p, rng);
    const double tgt = spec_uniform(rng) * sum;
    double acc = 0.0;
    for (const auto& kv : diffs) {
        acc += kv.second;
        if (acc >= tgt) return kv.first;
    }
    return diffs.back().first;
}


// One verify round with a deterministic drafter (proposal q = point mass at drafts[j]). `dist_of(j, SpecDist&)` supplies the target distribution of row j
// (rows 0..nd; row j is the distribution of the token that follows drafts[0..j-1]). Draft j is accepted with probability p_j(drafts[j]); on rejection the
// token is drawn from p_j with the draft removed (the residual max(0, p - q) / norm for a point-mass q). If all nd drafts are accepted, a bonus token is drawn from
// row nd. Writes seq[0..m] (m accepted drafts + one more token), returns m. Lossless: the output sequence has exactly the target distribution.
template <class DistFn>
inline int spec_round_point(int nd, const int* drafts, DistFn&& dist_of, std::mt19937_64& rng, int* seq, int* n_acc = nullptr) {
    int acc = 0;
    for (int j = 0; j <= nd; ++j) {
        SpecDist pj;
        dist_of(j, pj);
        if (j < nd) {
            SpecDist q;
            q.sp.assign(1, {drafts[j], 1.0f});
            if (spec_accept(pj, q, drafts[j], rng)) { seq[j] = drafts[j]; acc = j + 1; continue; }
            seq[j] = sample_residual(pj, q, rng);
        } else {
            seq[j] = sample_dist(pj, rng);
        }
        if (n_acc) *n_acc = acc;
        return j;
    }
    return nd;  // unreachable
}

}  // namespace qv
