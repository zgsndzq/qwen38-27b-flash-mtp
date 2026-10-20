#include "llama-perf.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace llama_perf {

namespace {

std::once_flag g_init_once;
std::atomic<int> g_level{-1};   // -1 = not resolved yet
std::mutex g_mu;
std::FILE * g_fp = nullptr;
bool g_owned = false;           // true when we opened the file (and must close it)
double g_t0 = 0.0;

// Accumulators between two note_decode calls. decode is single-threaded per context,
// so a plain pair of counters is enough; take() drains them.
int    g_graphs    = 0;
double g_graph_ms  = 0.0;
int    g_splits    = 0;   // total sched splits across those graphs

void init_once() {
    const char * env = std::getenv("LLAMA_PERF_LOG");
    int level = 0;
    if (env != nullptr && env[0] != '\0') {
        level = std::atoi(env);
        if (level < 0) {
            level = 0;
        }
    }
    g_level.store(level, std::memory_order_relaxed);
    if (level == 0) {
        return;
    }

    const char * path = std::getenv("LLAMA_PERF_LOG_FILE");
    if (path != nullptr && path[0] != '\0') {
        g_fp = std::fopen(path, "w");
    }
    if (g_fp == nullptr) {
        g_fp = stderr;
    } else {
        g_owned = true;
    }

    std::fprintf(g_fp, "# llama_perf level=%d\n", level);
    std::fprintf(g_fp, "t_ms,kind,ctx,n_tokens,max_pos,ms,acc,graphs,graph_ms,splits\n");
    std::fflush(g_fp);

    g_t0 = std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void ensure() {
    if (g_level.load(std::memory_order_relaxed) < 0) {
        std::call_once(g_init_once, init_once);
    }
}

} // namespace

bool enabled() {
    ensure();
    return g_level.load(std::memory_order_relaxed) > 0;
}

int level() {
    ensure();
    return g_level.load(std::memory_order_relaxed);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void note_graph(double ms, int splits) {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_graphs;
    g_graph_ms += ms;
    g_splits   += splits;
}

int graph_count_take() {
    std::lock_guard<std::mutex> lock(g_mu);
    const int v = g_graphs;
    g_graphs = 0;
    return v;
}

double graph_ms_take() {
    std::lock_guard<std::mutex> lock(g_mu);
    const double v = g_graph_ms;
    g_graph_ms = 0.0;
    return v;
}

int graph_splits_take() {
    std::lock_guard<std::mutex> lock(g_mu);
    const int v = g_splits;
    g_splits = 0;
    return v;
}

void note_decode(int ctx_type, int n_tokens, int max_pos, double ms, int n_graphs, double graph_ms, int n_splits) {
    const int lvl = level();
    if (lvl < 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_fp == nullptr) {
        return;
    }
    std::fprintf(g_fp, "%.3f,D,%d,%d,%d,%.3f,,%d,%.3f,%d\n",
                 now_ms() - g_t0, ctx_type, n_tokens, max_pos, ms, n_graphs, graph_ms, n_splits);
    std::fflush(g_fp);   // the monitor tails this file; a buffered line is a missing line
}

void note_accept(int n_accepted) {
    const int lvl = level();
    if (lvl < 2) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_fp == nullptr) {
        return;
    }
    std::fprintf(g_fp, "%.3f,A,0,0,0,0.000,%d,0,0.000\n", now_ms() - g_t0, n_accepted);
    std::fflush(g_fp);
}

} // namespace llama_perf
