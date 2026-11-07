// Pipeline compile journal and quarantine (pipe_journal.c).
//
// A pipeline that crashes the driver's shader compiler (an access violation inside nvgpucomp64.dll, say) ends the
// process, and before this the next run compiled the same pipeline at the same place and crashed again. The journal
// names the pipelines being compiled in a small file next to the pipeline cache, written before and after each
// vkCreate*Pipelines; the OS keeps what was written when the process dies. The crash handler marks the pipeline the
// faulting thread was compiling. At the next boot a marked pipeline gets a strike in pipeline-quarantine.txt:
//   strike 1  it is built with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT, another path through the driver's compiler;
//   strike 2  it is not built: the draws or dispatches that need it are skipped.
// Deleting pipeline-quarantine.txt clears every strike. TOTK_PIPE_QUARANTINE=0 turns the journal and the strikes off.
#pragma once
#include <stdint.h>

enum { PJ_GRAPHICS = 1, PJ_COMPUTE = 2 };

// dir: the folder of pipeline-cache.bin. Reads the journal the previous run left, adds its strikes to the quarantine
// and starts an empty journal. Call once, before the first pj_begin; on = 0 leaves every other call a no-op.
void pj_open(const char *dir, int on);
// Strikes of a pipeline id: 0 build as usual, 1 build without optimization, 2 or more do not build.
int pj_strikes(uint64_t id);
// Around one vkCreate*Pipelines on this thread: pj_begin returns a slot for pj_end (-1 when the journal is off or full).
int pj_begin(uint64_t id, int kind, int strikes);
void pj_end(int slot);
// From the crash handler, on the faulting thread. module: the module the fault happened in, NULL for this executable.
// Marks the pipeline this thread was compiling; after a fault in another module with no compile on this thread, marks
// every pipeline being compiled (the driver may compile on a thread of its own). Logs what it marked.
void pj_crash(const char *module);
