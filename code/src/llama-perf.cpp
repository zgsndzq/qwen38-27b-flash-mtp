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
    std::fprintf(g_fp, "t_ms,kind,ctx,n_tokens,max_pos,ms,acc\n");
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

void note_decode(int ctx_type, int n_tokens, int max_pos, double ms) {
    const int lvl = level();
    if (lvl < 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_fp == nullptr) {
        return;
    }
    std::fprintf(g_fp, "%.3f,D,%d,%d,%d,%.3f,\n", now_ms() - g_t0, ctx_type, n_tokens, max_pos, ms);
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
    std::fprintf(g_fp, "%.3f,A,0,0,0,0.000,%d\n", now_ms() - g_t0, n_accepted);
    std::fflush(g_fp);
}

} // namespace llama_perf
