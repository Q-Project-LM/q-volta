// qv_server: llama-server compatible HTTP server for Q-Volta engines (CUDA, CPU, or the scripted mock engine for tests).
#include <signal.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#include "server/app.h"

#ifndef QV_GIT_REV
#define QV_GIT_REV "dev"
#endif

using namespace qv::server;
using Clock = std::chrono::steady_clock;

namespace {
std::atomic<qv::http::Server*> g_server{nullptr};
void on_signal(int) { if (auto* s = g_server.load()) s->stop(); }

bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return (bool)f; }

void print_help() {
    std::printf(
        "usage: qv_server -m MODEL [options]\n\n"
        "qv_server is a llama-server compatible HTTP server. Every llama-server flag spelling and LLAMA_ARG_* environment variable\n"
        "of llama.cpp 9d6f04f is accepted; flags without a meaning for Q-Volta are warned about and ignored (see docs/SERVER_COMPAT.md).\n\n"
        "common flags:\n"
        "  -m, --model PATH            Q-Volta model (PREFIX or PREFIX.qv)\n"
        "  --models-preset PATH        INI presets file with [*] defaults and [model-id] sections (router mode)\n"
        "  -c, --ctx-size N            context size (default 4096; per slot, all slots share one engine)\n"
        "  -t, --threads N             CPU threads (default min(32, cores))\n"
        "  -ngl, --n-gpu-layers N      0 = CPU engine, >0 / auto / all = CUDA engine when a device exists (all-or-nothing)\n"
        "  -b / -ub N                  prefill chunk size (-ub wins)\n"
        "  --host H --port P           default 127.0.0.1:9931 (llama-server default of the reference commit)\n"
        "  --api-key K[,K2] / --api-key-file F, -a/--alias NAME, -np/--parallel N, --cache-prompt/--no-cache-prompt\n"
        "  --jinja/--no-jinja, --chat-template NAME, --chat-template-kwargs JSON, --reasoning-format auto|none|deepseek|deepseek-legacy\n"
        "  --reasoning on|off|auto, --metrics, --slots/--no-slots, --props, --slot-save-path DIR, --path DIR, --no-webui\n"
        "  -n/--predict, --temp, --top-k, --top-p, --min-p, --typical, --repeat-penalty, --repeat-last-n, --presence-penalty,\n"
        "  --frequency-penalty, --mirostat, --xtc-*, --dry-*, --dynatemp-*, -s/--seed, --samplers, --ignore-eos, -l/--logit-bias,\n"
        "  --grammar, --json-schema, -to/--timeout, --cors-*, --api-prefix, --log-disable, -v\n\n"
        "Q-Volta flags:\n"
        "  --backend cuda|cpu|auto|mock   --kv fp32|fp16|int4   --mtp / --mtp-depth N / --mtp-max-ctx N (plain decode past N tokens, default 65536, 0 = never) / --mtp-file PATH (Inkling)   --prefill-chunk N   --tokenizer PATH\n"
        "  --max-queue N (503 beyond N waiting requests; default unbounded like llama-server)\n"
        "  mock backend (tests): --mock-text TEXT --mock-loop --mock-delay-us N (env QV_MOCK_LOAD_MS delays loading)\n");
}
}  // namespace

int main(int argc, char** argv) {
    Options o;
    bool exit_now = false;
    int rc = parse_args(argc, argv, o, exit_now);
    if (exit_now) {
        if (rc == -1) print_help();
        else std::printf("version: qv-%s (qv_server, llama-server compatible; reference llama.cpp 9d6f04f)\n", QV_GIT_REV);
        return 0;
    }
    if (rc) return rc;
    for (auto& w : o.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    signal(SIGPIPE, SIG_IGN);

    auto app = std::make_unique<App>();
    app->build_info = std::string("qv-") + QV_GIT_REV;
    if (o.ctx <= 0) o.ctx = 4096;
    app->n_ctx = o.ctx;
    app->o = o;
    app->base_opts = o;
    app->model_id = !o.alias.empty() ? o.alias[0] : (o.model.empty() ? std::string("qv-mock") : o.model);
    app->started_at = (long long)std::time(nullptr);
    app->init_slots();
    std::string host = o.host;
    if (auto c = host.find(','); c != std::string::npos) host = host.substr(0, c);

    App* ap = app.get();
    const int max_conns = o.http_threads > 0 ? std::max(o.http_threads, ap->n_slots + 2) : 256;
    qv::http::Server srv(host, o.port, max_conns, [ap](const qv::http::Request& r, qv::http::Conn& c) { ap->route(r, c); });
    srv.set_timeout(o.timeout);
    std::string err;
    if (!srv.listen_on(err)) { std::fprintf(stderr, "couldn't bind HTTP server socket, hostname: %s, port: %d (%s)\n", host.c_str(), o.port, err.c_str()); return 1; }
    g_server = &srv;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    std::thread http_thread([&] { srv.run(); });
    std::fprintf(stderr, "[qv_server] listening on http://%s:%d (loading model)\n", host.c_str(), srv.port());

    // ---- model loading (the HTTP port is already up: /health answers 503 "Loading model" until ready)
    auto t0 = Clock::now();
    auto fail = [&](const std::string& m) {
        std::fprintf(stderr, "error: %s\n", m.c_str());
        srv.stop();
        http_thread.join();
        return 1;
    };
    if (!o.models_preset.empty()) {
        std::vector<std::string> pw;
        try {
            app->presets = load_presets_ini(o.models_preset, o, pw);
        } catch (const std::exception& e) {
            return fail(std::string("models-preset: ") + e.what());
        }
        for (const auto& w : pw) std::fprintf(stderr, "warning: %s\n", w.c_str());
        const ModelPreset* start_p = nullptr;
        if (!o.model.empty()) start_p = app->find_preset(o.model);
        if (!start_p) {
            for (const auto& p : app->presets) if (p.opts.preload) { start_p = &p; break; }
        }
        auto has_file = [](const std::string& p) {
            if (p.empty()) return false;
            std::string s = p;
            if (s.size() > 3 && s.compare(s.size() - 3, 3, ".qv") == 0 && !file_exists(s)) s.resize(s.size() - 3);
            return file_exists(p) || file_exists(s) || file_exists(s + ".qv") || file_exists(s + ".qv.json");
        };
        if (start_p && (start_p->opts.backend == "mock" || has_file(start_p->opts.model))) {
            try {
                app->load_model(start_p->opts, start_p->id);
            } catch (const std::exception& e) {
                return fail(std::string("model load failed: ") + e.what());
            }
            std::fprintf(stderr, "[qv_server] preset '%s' preloaded in %.1fs (%zu presets total; backend=%s, n_ctx=%d, slots=%d, mtp=%d, tokenizer=%s)\n",
                         start_p->id.c_str(), std::chrono::duration<double>(Clock::now() - t0).count(), app->presets.size(),
                         app->backend_name.c_str(), app->n_ctx, app->n_slots, (int)app->o.mtp, app->tokenizer_path.c_str());
        } else if (start_p) {
            app->model_id.clear();
            std::fprintf(stderr, "[qv_server] warning: preloaded preset '%s' model '%s' not found on disk; waiting for requests\n",
                         start_p->id.c_str(), start_p->opts.model.c_str());
        } else {
            app->model_id.clear();
            std::fprintf(stderr, "[qv_server] router ready with %zu presets (no model preloaded; waiting for requests)\n",
                         app->presets.size());
        }
    } else {
        try {
            app->load_model(o);
        } catch (const std::exception& e) {
            return fail(std::string("model load failed: ") + e.what());
        }
        std::fprintf(stderr, "[qv_server] model loaded in %.1fs (backend=%s, threads=%d, n_ctx=%d, slots=%d, mtp=%d, tokenizer=%s)\n",
                     std::chrono::duration<double>(Clock::now() - t0).count(), app->backend_name.c_str(),
                     app->o.threads > 0 ? app->o.threads : (int)std::min(32u, std::max(1u, std::thread::hardware_concurrency())),
                     app->n_ctx, app->n_slots, (int)app->o.mtp, app->tokenizer_path.c_str());
    }
    app->ready = true;
    app->start_sleep_watcher();
    std::fprintf(stderr, "[qv_server] server is listening on http://%s:%d (model id \"%s\")\n", host.c_str(), srv.port(), app->model_id.c_str());
    http_thread.join();
    app->stop_sleep_watcher();
    g_server = nullptr;
    std::fprintf(stderr, "[qv_server] shut down\n");
    return 0;
}
