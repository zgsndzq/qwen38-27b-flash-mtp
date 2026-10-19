#pragma once

// Decode instrumentation, deliberately kept outside the engine code: one header,
// one translation unit, and a handful of call sites that do nothing when it is off.
//
//   LLAMA_PERF_LOG=0   off (default)
//   LLAMA_PERF_LOG=1   one line per llama_decode call
//   LLAMA_PERF_LOG=2   also one line per speculative accept
//   LLAMA_PERF_LOG_FILE=<path>   where to write (default: stderr)
//
// Output is CSV, one record per line, so tools/perf_monitor.py can tail it live and
// anything else can just load it. Enable it from features.ini; a release build simply
// leaves the keys out and pays nothing beyond one getenv at first use.

#include <cstdint>

#include "llama.h"   // LLAMA_API: the symbols live in llama.dll, common/ imports them

namespace llama_perf {

// true when LLAMA_PERF_LOG > 0; resolved once, safe to call per decode
LLAMA_API bool enabled();

// 0 = off, 1 = decode lines, 2 = decode + accept lines
LLAMA_API int level();

// monotonic milliseconds, for callers that time a region themselves
LLAMA_API double now_ms();

// one record per llama_decode: ctx_type is llama_context_type as int (0 = target,
// 1 = MTP/draft), n_tokens is the batch size (1 + drafts for a verify round),
// max_pos is the highest position in the batch, ms is the wall time of the call.
LLAMA_API void note_decode(int ctx_type, int n_tokens, int max_pos, double ms);

// one record per speculative accept: how many drafted tokens the target accepted
LLAMA_API void note_accept(int n_accepted);

} // namespace llama_perf
