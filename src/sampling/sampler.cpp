#include "sampling/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace qv {

Sampler::Sampler(const SamplingParams& p, const std::vector<int>& prompt) : p_(p), hist_(prompt) {
    uint32_t seed = p.has_seed ? (uint32_t)p.seed : (uint32_t)(std::random_device{}() ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    rng_.seed(seed);
    mu_ = 2.0f * p.mirostat_tau;
}

void Sampler::accept(int tok) {
    hist_.push_back(tok);
    if (constraint_) constraint_->accept(tok);
}

void Sampler::apply_penalties(std::vector<float>& lg) const {
    if (p_.repeat_penalty == 1.0f && p_.frequency_penalty == 0 && p_.presence_penalty == 0) return;
    size_t n = hist_.size();
    size_t last = p_.repeat_last_n < 0 ? n : std::min<size_t>(n, (size_t)p_.repeat_last_n);
    if (last == 0) return;
    std::unordered_map<int, int> cnt;
    cnt.reserve(last);
    for (size_t i = n - last; i < n; ++i) ++cnt[hist_[i]];
    for (auto& [t, c] : cnt) {
        if (t < 0 || t >= (int)lg.size()) continue;
        if (p_.repeat_penalty != 1.0f) lg[t] = lg[t] <= 0 ? lg[t] * p_.repeat_penalty : lg[t] / p_.repeat_penalty;
        lg[t] -= (float)c * p_.frequency_penalty + (c > 0 ? 1.0f : 0.0f) * p_.presence_penalty;
    }
}

// DRY (don't repeat yourself). Follows llama_sampler_dry semantics: a token that would extend a sequence already seen in the
// context (suffix match length >= allowed_length) is penalised by multiplier * base^(len - allowed_length).
// Simplified matching (direct suffix comparison, match length capped at 64; sequence breakers act on single tokens).
void Sampler::apply_dry(std::vector<float>& lg) const {
    if (p_.dry_multiplier <= 0 || p_.dry_base < 1.0f || hist_.size() < 2) return;
    size_t n = hist_.size();
    size_t win = p_.dry_penalty_last_n < 0 ? n : std::min<size_t>(n, (size_t)p_.dry_penalty_last_n);
    size_t beg = n - win;
    std::vector<int> br;
    for (auto& s : p_.dry_breaker_tokens) if (!s.empty()) br.push_back(s[0]);
    auto is_br = [&](int t) { return std::find(br.begin(), br.end(), t) != br.end(); };
    // the matched suffix must not contain a breaker: find the last breaker in the window (the last token itself is excluded from the check)
    size_t start = beg;
    for (size_t i = n - 1; i-- > beg;) if (is_br(hist_[i])) { start = i + 1; break; }
    if (is_br(hist_[n - 1])) return;
    std::unordered_map<int, int> best;
    const int kMax = 64;
    for (size_t i = beg + 1; i < n; ++i) {
        // would hist_[i] follow the suffix ending at n-1? compare hist_[i-1-k] with hist_[n-1-k]
        int len = 0;
        while (len < kMax && i >= beg + 1 + (size_t)len && n - 1 >= (size_t)len + start && hist_[i - 1 - len] == hist_[n - 1 - len] && i - 1 - len >= beg) ++len;
        if (len >= p_.dry_allowed_length && !is_br(hist_[i])) {
            int& b = best[hist_[i]];
            b = std::max(b, len);
        }
    }
    for (auto& [t, len] : best)
        if (t >= 0 && t < (int)lg.size()) lg[t] -= p_.dry_multiplier * std::pow(p_.dry_base, (float)(len - p_.dry_allowed_length));
}

void Sampler::sort_desc() {
    if (sorted_) return;
    std::sort(cur_.begin(), cur_.end(), [](const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); });
    sorted_ = true;
}

void Sampler::keep_best_one() {
    if (cur_.empty()) return;
    if (!sorted_) {
        auto it = std::min_element(cur_.begin(), cur_.end(), [](const Cand& a, const Cand& b) {
            return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
        });
        cur_[0] = *it;
        sorted_ = true;
    }
    cur_.resize(1);
}

void Sampler::softmax() {
    sort_desc();
    if (cur_.empty()) return;
    float mx = cur_[0].logit;
    double s = 0;
    for (auto& c : cur_) { c.p = std::exp(c.logit - mx); s += c.p; }
    for (auto& c : cur_) c.p = (float)(c.p / s);
}

void Sampler::s_top_k(int k) {
    if (k <= 0 || k >= (int)cur_.size()) return;
    k = std::max(k, p_.min_keep > 0 ? p_.min_keep : 1);
    auto gt = [](const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); };
    if (!sorted_) {
        std::nth_element(cur_.begin(), cur_.begin() + k, cur_.end(), gt);
        cur_.resize(k);
        std::sort(cur_.begin(), cur_.end(), gt);
        sorted_ = true;
    } else cur_.resize(k);
}

void Sampler::s_top_p(float p) {
    if (p >= 1.0f) return;
    if (!sorted_ && cur_.size() > 1024) {
        float mx = -INFINITY;
        for (const auto& c : cur_) mx = std::max(mx, c.logit);
        if (std::isfinite(mx)) {
            double sum_exp = 0.0;
            std::vector<Cand> top;
            top.reserve(256);
            for (const auto& c : cur_) {
                const float d = c.logit - mx;
                if (d > -36.0f) sum_exp += std::exp((double)d);
                if (d >= -14.0f) top.push_back(c);
            }
            if (sum_exp > 0.0 && std::isfinite(sum_exp) && !top.empty()) {
                std::sort(top.begin(), top.end(), [](const Cand& a, const Cand& b) {
                    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
                });
                const size_t min_k = (size_t)std::max(1, p_.min_keep);
                double cum = 0.0;
                for (size_t i = 0; i < top.size(); ++i) {
                    top[i].p = (float)(std::exp((double)top[i].logit - (double)mx) / sum_exp);
                    cum += top[i].p;
                    if (cum >= p && i + 1 >= min_k) {
                        top.resize(i + 1);
                        cur_.swap(top);
                        sorted_ = true;
                        return;
                    }
                }
            }
        }
    }
    softmax();
    double c = 0;
    size_t keep = cur_.size();
    for (size_t i = 0; i < cur_.size(); ++i) {
        c += cur_[i].p;
        if (c >= p && i + 1 >= (size_t)std::max(1, p_.min_keep)) { keep = i + 1; break; }
    }
    cur_.resize(keep);
}

void Sampler::s_min_p(float p) {
    if (p <= 0.0f || cur_.empty()) return;
    if (!sorted_ && cur_.size() > 1024) {
        float mx = -INFINITY;
        for (const auto& c : cur_) mx = std::max(mx, c.logit);
        if (std::isfinite(mx)) {
            const float thr = mx + std::log(p) - 1e-4f;
            std::vector<Cand> top;
            top.reserve(256);
            for (const auto& c : cur_) {
                if (c.logit >= thr) top.push_back(c);
            }
            const size_t min_k = (size_t)std::max(1, p_.min_keep);
            if (top.size() >= min_k) {
                std::sort(top.begin(), top.end(), [](const Cand& a, const Cand& b) {
                    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
                });
                size_t keep = 0;
                while (keep < top.size() && std::exp((double)top[keep].logit - (double)mx) >= (double)p) ++keep;
                keep = std::max(keep, min_k);
                top.resize(std::min(keep, top.size()));
                cur_.swap(top);
                sorted_ = true;
                return;
            }
            s_top_k((int)min_k);
            return;
        }
    }
    softmax();
    float thr = cur_[0].p * p;
    size_t keep = 0;
    while (keep < cur_.size() && cur_[keep].p >= thr) ++keep;
    keep = std::max<size_t>(keep, std::max(1, p_.min_keep));
    cur_.resize(std::min(keep, cur_.size()));
}

void Sampler::s_typical(float p) {
    if (p >= 1.0f || cur_.empty()) return;
    softmax();
    double ent = 0;
    for (auto& c : cur_) if (c.p > 0) ent += -c.p * std::log(c.p);
    std::vector<std::pair<float, size_t>> sh(cur_.size());
    for (size_t i = 0; i < cur_.size(); ++i) sh[i] = {(float)std::fabs(-std::log(std::max(cur_[i].p, 1e-30f)) - ent), i};
    std::stable_sort(sh.begin(), sh.end(), [](auto& a, auto& b) { return a.first < b.first; });
    double cum = 0;
    size_t last = sh.size();
    for (size_t i = 0; i < sh.size(); ++i) {
        cum += cur_[sh[i].second].p;
        if (cum > p && i + 1 >= (size_t)std::max(1, p_.min_keep)) { last = i + 1; break; }
    }
    std::vector<char> keep(cur_.size(), 0);
    for (size_t i = 0; i < last; ++i) keep[sh[i].second] = 1;
    std::vector<Cand> out;
    for (size_t i = 0; i < cur_.size(); ++i) if (keep[i]) out.push_back(cur_[i]);
    cur_ = std::move(out);
}

void Sampler::s_top_n_sigma(float n) {
    if (n <= 0.0f || cur_.size() < 2) return;
    float mx = -INFINITY;
    double sum = 0;
    for (auto& c : cur_) { mx = std::max(mx, c.logit); sum += c.logit; }
    double mean = sum / cur_.size(), var = 0;
    for (auto& c : cur_) var += (c.logit - mean) * (c.logit - mean);
    double sd = std::sqrt(var / cur_.size());
    float thr = (float)(mx - n * sd);
    std::vector<Cand> out;
    for (auto& c : cur_) if (c.logit >= thr) out.push_back(c);
    cur_ = std::move(out);
}

void Sampler::s_xtc() {
    if (p_.xtc_probability <= 0.0f || p_.xtc_threshold > 0.5f || cur_.size() < 2) return;
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    if (d(rng_) > p_.xtc_probability) return;
    softmax();
    size_t pos_last = 0;
    for (size_t i = 0; i < cur_.size(); ++i) if (cur_[i].p >= p_.xtc_threshold) pos_last = i; else break;
    if (pos_last >= 1 && cur_.size() - pos_last >= (size_t)std::max(1, p_.min_keep)) cur_.erase(cur_.begin(), cur_.begin() + pos_last);
}

void Sampler::s_temp() {
    float t = p_.temperature;
    if (p_.dynatemp_range > 0.0f && cur_.size() > 1) {
        float mn = std::max(0.0f, t - p_.dynatemp_range), mxt = t + p_.dynatemp_range;
        softmax();
        double ent = 0;
        for (auto& c : cur_) if (c.p > 0) ent -= c.p * std::log(c.p);
        double maxent = std::log((double)cur_.size()), norm = maxent > 0 ? ent / maxent : 0;
        t = (float)(mn + (mxt - mn) * std::pow(norm, p_.dynatemp_exponent));
    }
    if (t <= 0.0f) { keep_best_one(); return; }
    for (auto& c : cur_) c.logit /= t;
}

int Sampler::draw() {
    if (cur_.empty()) return 0;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (cur_.size() == 1) { (void)dist(rng_); return cur_[0].id; }
    float mx = cur_[0].logit;
    if (!sorted_) for (auto& c : cur_) mx = std::max(mx, c.logit);
    if (!std::isfinite(mx)) {
        (void)dist(rng_);
        return cur_[0].id;
    }
    double sum = 0;
    for (auto& c : cur_) {
        const float d = c.logit - mx;
        if (d > -36.0f) {
            c.p = expf(d);
            sum += c.p;
        } else {
            c.p = 0.0f;
        }
    }
    double rnd = dist(rng_), run = 0, tgt = sum * rnd;
    int sel = (int)cur_.size() - 1;
    for (size_t i = 0; i < cur_.size(); ++i) { run += cur_[i].p; if (run >= tgt) { sel = (int)i; break; } }
    if (want_post_ && sum > 0.0) {
        const double inv_s = 1.0 / sum;
        for (auto& c : cur_) c.p = (float)(c.p * inv_s);
    }
    return cur_[sel].id;
}

int Sampler::mirostat_v2() {
    softmax();
    // truncate tokens with surprise > mu
    size_t keep = cur_.size();
    for (size_t i = 0; i < cur_.size(); ++i)
        if (-std::log2(std::max(cur_[i].p, 1e-30f)) > mu_) { keep = std::max<size_t>(1, i); break; }
    cur_.resize(keep);
    softmax();
    int id = draw();
    float pp = 1e-30f;
    for (auto& c : cur_) if (c.id == id) pp = std::max(c.p, 1e-30f);
    mu_ -= p_.mirostat_eta * (-std::log2(pp) - p_.mirostat_tau);
    return id;
}

int Sampler::mirostat_v1() {
    const int m = 100;
    softmax();
    double N = (double)cur_.size(), sum_ti_bi = 0, sum_ti_sq = 0;
    for (size_t i = 0; i + 1 < std::min<size_t>(m, cur_.size()); ++i) {
        double t = std::log((double)(i + 2) / (double)(i + 1)), b = std::log2(std::max(cur_[i].p, 1e-30f) / std::max(cur_[i + 1].p, 1e-30f));
        sum_ti_bi += t * b;
        sum_ti_sq += t * t;
    }
    double s_hat = sum_ti_sq > 0 ? sum_ti_bi / sum_ti_sq : 0;
    double eps_hat = s_hat - 1, k = std::pow((eps_hat * std::pow(2.0, mu_)) / (1 - std::pow(N, -eps_hat)), 1 / s_hat);
    s_top_k(std::isfinite(k) && k >= 1 ? (int)std::min(k, 1e9) : 1);
    softmax();
    int id = draw();
    float pp = 1e-30f;
    for (auto& c : cur_) if (c.id == id) pp = std::max(c.p, 1e-30f);
    mu_ -= p_.mirostat_eta * (-std::log2(pp) - p_.mirostat_tau);
    return id;
}

float Sampler::last_prob(int tok) const {
    for (auto& [id, p] : last_cands_) if (id == tok) return p;
    return 0.0f;
}

int Sampler::sample(std::vector<float>& lg) {
    for (auto& [t, b] : p_.logit_bias) if (t >= 0 && t < (int)lg.size()) lg[t] += b;
    if (constraint_) constraint_->apply(lg);
    const int V = (int)lg.size();
    auto finish = [&](int id) {
        last_cands_.clear();
        if (want_post_) {
            softmax();
            for (auto& c : cur_) last_cands_.emplace_back(c.id, c.p);
        }
        return id;
    };
    cur_.clear();
    sorted_ = false;
    if (p_.mirostat != 0) {
        apply_penalties(lg);
        cur_.resize(V);
        for (int i = 0; i < V; ++i) cur_[i] = {i, lg[i], 0};
        s_temp();
        int id = p_.mirostat == 1 ? mirostat_v1() : mirostat_v2();
        return finish(id);
    }
    bool pen_done = false;
    auto ensure = [&] {
        if (!cur_.empty()) return;
        if (constraint_ != nullptr && p_.top_n_sigma <= 0.0f) {
            cur_.reserve(std::min(V, 4096));
            for (int i = 0; i < V; ++i) {
                if (std::isfinite(lg[i])) cur_.push_back({i, lg[i], 0});
            }
            if (cur_.size() >= (size_t)std::max(1, p_.min_keep)) return;
            cur_.clear();
        }
        cur_.resize(V);
        for (int i = 0; i < V; ++i) cur_[i] = {i, lg[i], 0};
    };
    for (const std::string& s : p_.samplers) {
        if (s == "penalties") { apply_penalties(lg); pen_done = true; }
        else if (s == "dry") apply_dry(lg);
        else {
            ensure();
            if (s == "top_n_sigma") s_top_n_sigma(p_.top_n_sigma);
            else if (s == "top_k") s_top_k(p_.top_k);
            else if (s == "typ_p") s_typical(p_.typical_p);
            else if (s == "top_p") s_top_p(p_.top_p);
            else if (s == "min_p") s_min_p(p_.min_p);
            else if (s == "xtc") s_xtc();
            else if (s == "temperature") s_temp();
        }
    }
    (void)pen_done;
    ensure();
    if (p_.temperature <= 0.0f && std::find(p_.samplers.begin(), p_.samplers.end(), "temperature") == p_.samplers.end()) {
        keep_best_one();
    }
    int id = draw();
    return finish(id);
}

TokenLogprob Sampler::logprob(const std::vector<float>& raw, int chosen, int top_n) {
    float mx = *std::max_element(raw.begin(), raw.end());
    double s = 0;
    for (float x : raw) s += std::exp((double)x - mx);
    const double lse = mx + std::log(s);
    TokenLogprob r;
    r.token = chosen;
    r.logprob = (float)(raw[chosen] - lse);
    if (top_n > 0) {
        std::vector<int> idx(raw.size());
        std::iota(idx.begin(), idx.end(), 0);
        top_n = std::min<int>(top_n, (int)raw.size());
        std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(), [&](int a, int b) { return raw[a] > raw[b] || (raw[a] == raw[b] && a < b); });
        for (int i = 0; i < top_n; ++i) r.top.emplace_back(idx[i], (float)(raw[idx[i]] - lse));
    }
    return r;
}

}  // namespace qv
