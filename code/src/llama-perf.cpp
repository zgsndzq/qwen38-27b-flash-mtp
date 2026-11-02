#include "llama-perf.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

#ifdef _WIN32
#   include <io.h>
#else
#   include <unistd.h>
#endif

namespace llama_perf {

namespace {

std::once_flag g_init_once;
std::atomic<int> g_level{-1};   // -1 = not resolved yet
std::mutex g_mu;
std::FILE * g_fp = nullptr;
bool g_owned = false;           // true when we opened the file (and must close it)
size_t g_max = 1000;            // LLAMA_PERF_LOG_MAX, 0 = no cap
std::deque<long> g_off;         // start offset of every record still in the file
std::string g_header;           // rewritten at the top after every rotate
double g_t0 = 0.0;

// Accumulators between two note_decode calls. decode is single-threaded per context,
// so a plain pair of counters is enough; take() drains them.
int    g_graphs    = 0;
double g_graph_ms  = 0.0;
int    g_splits    = 0;   // total sched splits across those graphs

void set_file_size(std::FILE * fp, long size) {
#ifdef _WIN32
    _chsize_s(_fileno(fp), size);
#else
    (void) !::ftruncate(::fileno(fp), size);
#endif
}

// Drop the older half of the file, so a long run keeps recent records instead of
// freezing on the first N. Called with g_mu held.
void rotate() {
    const size_t half  = g_off.size() / 2;
    const long   start = g_off[half];

    std::fseek(g_fp, start, SEEK_SET);
    std::string keep;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, g_fp)) > 0) {
        keep.append(buf, n);
    }

    std::fseek(g_fp, 0, SEEK_SET);
    std::fwrite(g_header.data(), 1, g_header.size(), g_fp);
    std::fwrite(keep.data(), 1, keep.size(), g_fp);
    set_file_size(g_fp, (long) (g_header.size() + keep.size()));
    std::fseek(g_fp, 0, SEEK_END);

    for (size_t i = 0; i < half; ++i) {
        g_off.pop_front();
    }
    for (size_t i = 0; i < g_off.size(); ++i) {
        g_off[i] = g_off[i] - start + (long) g_header.size();
    }
}

// One record; capped so the file cannot grow without bound. Caller holds no lock.
void emit(const char * fmt, ...) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_fp == nullptr) {
        return;
    }
    if (g_max != 0 && g_off.size() >= g_max) {
        if (!g_owned) {
            return;   // stderr cannot be rewound, just stop
        }
        rotate();
    }
    g_off.push_back(g_owned ? std::ftell(g_fp) : 0);

    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_fp, fmt, ap);
    va_end(ap);
    std::fflush(g_fp);   // the monitor tails this file; a buffered line is a missing line
}

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

    const char * cap = std::getenv("LLAMA_PERF_LOG_MAX");
    if (cap != nullptr && cap[0] != '\0') {
        const long v = std::atol(cap);
        g_max = v > 0 ? (size_t) v : 0;
    }

    const char * path = std::getenv("LLAMA_PERF_LOG_FILE");
    if (path != nullptr && path[0] != '\0') {
        g_fp = std::fopen(path, "w+");   // w+: rotate() reads the tail back
    }
    if (g_fp == nullptr) {
        g_fp = stderr;
    } else {
        g_owned = true;
    }

    char hdr[128];
    std::snprintf(hdr, sizeof hdr, "# llama_perf level=%d max=%zu\n", level, g_max);
    g_header  = hdr;
    g_header += "t_ms,kind,ctx,n_tokens,max_pos,ms,acc,graphs,graph_ms,splits\n";
    std::fwrite(g_header.data(), 1, g_header.size(), g_fp);
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
    if (level() < 1) {
        return;
    }
    emit("%.3f,D,%d,%d,%d,%.3f,,%d,%.3f,%d\n",
         now_ms() - g_t0, ctx_type, n_tokens, max_pos, ms, n_graphs, graph_ms, n_splits);
}

void note_accept(int n_accepted) {
    if (level() < 2) {
        return;
    }
    emit("%.3f,A,0,0,0,0.000,%d,0,0.000\n", now_ms() - g_t0, n_accepted);
}

} // namespace llama_perf
