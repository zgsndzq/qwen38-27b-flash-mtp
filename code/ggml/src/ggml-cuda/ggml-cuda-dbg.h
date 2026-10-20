#pragma once

// Debug logs (MMQ dispatch choices, FA kernel routing) go to a file, never to the
// console: set GGML_CUDA_LOG_FILE=<path> to enable, leave it unset for silence.
// Inline + per-TU statics on purpose: mmq.cuh is included by many instance TUs and
// each may open the same path in append mode, which is fine for line-oriented logs.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

inline std::FILE * ggml_cuda_dbg_file() {
    static std::FILE * fp = nullptr;
    static std::once_flag flag;
    std::call_once(flag, [] {
        const char * p = std::getenv("GGML_CUDA_LOG_FILE");
        if (p && p[0]) {
            fp = std::fopen(p, "a");
        }
    });
    return fp;
}

inline void ggml_cuda_dbg_log(const char * fmt, ...) {
    // Sampling: per-op logs fire thousands of times a second, so the file is capped
    // at GGML_CUDA_LOG_MAX lines (default 5000) per process — plenty to observe and
    // judge, without ever growing unbounded.
    static int lines_left = [] {
        const char * e = std::getenv("GGML_CUDA_LOG_MAX");
        return e && atoi(e) > 0 ? atoi(e) : 5000;
    }();
    if (lines_left <= 0) {
        return;
    }
    --lines_left;
    std::FILE * fp = ggml_cuda_dbg_file();
    if (!fp) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(fp, fmt, ap);
    va_end(ap);
    std::fflush(fp);
}
