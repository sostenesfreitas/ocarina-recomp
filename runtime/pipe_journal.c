// Pipeline compile journal and quarantine: what and why in pipe_journal.h.
//
// pipeline-journal.bin: a Header, then SLOTS Slots. pj_begin fills a free slot and pj_end clears it, each with one
// positional write on an unbuffered stream, so the bytes are in the OS's file cache when the call returns and survive
// the process (not a power cut, which is fine for a crash record). The crash handler sets Header.crashed and the module,
// and flags the slots it blames. Only a run that recorded a crash gives strikes: a game closed from the task manager in
// the middle of a compile leaves slots in use and no crash, which is logged and forgiven.
//
// pipeline-quarantine.txt: one line per pipeline, "<id> <graphics|compute> <strikes>", read at boot and rewritten when
// a strike is added. Ids are pipe ids from vk_pipeline.c (pipe_journal_id, compute_journal_id): hashes of the state
// and of the shaders' input ids, the same across runs and translator versions.
#include "pipe_journal.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"

#define SLOTS 32
#define QMAX 1024
#define MAGIC "OOTPJRN1"
#define VERSION 1u

typedef struct {
    char magic[8];
    uint32_t version, slots;
    uint32_t crashed, pad;   // crashed: the crash handler ran (pj_crash)
    char module[64];         // the module it named, "" for this executable
} Header;
typedef struct { uint64_t id; uint32_t kind, flags; } Slot;
enum { IN_USE = 1, BLAMED = 2, NO_OPT = 4 };

static struct {
    int on;
    FILE *f;
    int lock;
    Slot slot[SLOTS];
    char qpath[1100];
    int qn;
    uint64_t qid[QMAX];
    uint8_t qkind[QMAX], qstrikes[QMAX];
} J;
static __thread int my_slot = -1;

static void lock(void) {
    while (__atomic_exchange_n(&J.lock, 1, __ATOMIC_ACQUIRE)) __builtin_ia32_pause();
}
static void unlock(void) { __atomic_store_n(&J.lock, 0, __ATOMIC_RELEASE); }

static const char *kind_name(uint32_t k) { return k == PJ_COMPUTE ? "compute" : "graphics"; }

// Caller holds the lock.
static void write_at(long off, const void *p, size_t n) {
    if (fseek(J.f, off, SEEK_SET) == 0) fwrite(p, n, 1, J.f);
}
static void write_slot(int i) { write_at((long)(sizeof(Header) + (size_t)i * sizeof(Slot)), &J.slot[i], sizeof(Slot)); }

// ---- quarantine file

static void quarantine_load(void) {
    FILE *f = fopen(J.qpath, "r");
    if (!f) return;
    char line[256], kind[16];
    unsigned long long id;
    unsigned strikes;
    while (fgets(line, sizeof line, f) && J.qn < QMAX) {
        if (line[0] == '#' || sscanf(line, "%llx %15s %u", &id, kind, &strikes) != 3 || !strikes) continue;
        J.qid[J.qn] = id, J.qkind[J.qn] = !strcmp(kind, "compute") ? PJ_COMPUTE : PJ_GRAPHICS;
        J.qstrikes[J.qn++] = (uint8_t)(strikes > 255 ? 255 : strikes);
    }
    fclose(f);
}

static void quarantine_save(void) {
    FILE *f = fopen(J.qpath, "w");
    if (!f) {
        rt_log("pipe_journal: cannot write %s", J.qpath);
        return;
    }
    fputs("# Ocarina Recomp: pipelines that crashed the GPU driver's shader compiler (runtime/pipe_journal.h).\n"
          "# <id> <kind> <strikes>. 1 strike: built without optimization. 2 or more: not built, what needs it is skipped.\n"
          "# Delete this file (or a line) to clear the strikes, after a driver update for instance.\n", f);
    for (int i = 0; i < J.qn; i++)
        fprintf(f, "%016llx %s %u\n", (unsigned long long)J.qid[i], kind_name(J.qkind[i]), J.qstrikes[i]);
    fclose(f);
}

static void quarantine_strike(uint64_t id, uint32_t kind) {
    for (int i = 0; i < J.qn; i++)
        if (J.qid[i] == id) {
            if (J.qstrikes[i] < 255) J.qstrikes[i]++;
            return;
        }
    if (J.qn == QMAX) return;
    J.qid[J.qn] = id, J.qkind[J.qn] = (uint8_t)kind, J.qstrikes[J.qn++] = 1;
}

// ---- the previous run's journal

static void previous_run(Header *h, const Slot *s) {
    // The boot prebuild compiles on every core, so a run can end with many compiles in progress: the ones the crash
    // named are listed, the others counted.
    int in_use = 0, blamed = 0;
    for (int i = 0; i < SLOTS; i++) {
        if (!(s[i].flags & IN_USE)) continue;
        in_use++;
        if (h->crashed && (s[i].flags & BLAMED))
            rt_log("pipe_journal: the previous run crashed while compiling %s pipeline %016llx%s", kind_name(s[i].kind),
                   (unsigned long long)s[i].id, s[i].flags & NO_OPT ? " (without optimization)" : "");
    }
    if (!h->crashed) {
        if (in_use) rt_log("pipe_journal: the previous run ended with %d compile(s) in progress and no crash recorded (closed or hung): no strike", in_use);
        return;
    }
    h->module[sizeof h->module - 1] = 0;
    for (int i = 0; i < SLOTS; i++)
        if ((s[i].flags & (IN_USE | BLAMED)) == (IN_USE | BLAMED)) quarantine_strike(s[i].id, s[i].kind), blamed++;
    rt_log("pipe_journal: the previous run crashed in %s; %d pipeline(s) get a strike, %d other compile(s) in progress do not",
           h->module[0] ? h->module : "the executable", blamed, in_use - blamed);
    if (blamed) quarantine_save();
}

void pj_open(const char *dir, int on) {
    if (!on || !dir || !*dir) {
        rt_log("pipe_journal: off%s", on ? " (no folder for the pipeline cache)" : " (TOTK_PIPE_QUARANTINE=0)");
        return;
    }
    char path[1100];
    snprintf(path, sizeof path, "%s/pipeline-journal.bin", dir);
    snprintf(J.qpath, sizeof J.qpath, "%s/pipeline-quarantine.txt", dir);
    quarantine_load();
    FILE *old = fopen(path, "rb");
    if (old) {
        Header h;
        Slot s[SLOTS];
        if (fread(&h, sizeof h, 1, old) == 1 && !memcmp(h.magic, MAGIC, 8) && h.version == VERSION && h.slots == SLOTS &&
            fread(s, sizeof *s, SLOTS, old) == SLOTS)
            previous_run(&h, s);
        fclose(old);
    }
    J.f = fopen(path, "wb+");
    if (!J.f) {
        rt_log("pipe_journal: cannot write %s; journal off", path);
        return;
    }
    setvbuf(J.f, NULL, _IONBF, 0);
    Header h = {.version = VERSION, .slots = SLOTS};
    memcpy(h.magic, MAGIC, 8);
    fwrite(&h, sizeof h, 1, J.f);
    fwrite(J.slot, sizeof(Slot), SLOTS, J.f);
    J.on = 1;
    int skip = 0, noopt = 0;
    for (int i = 0; i < J.qn; i++) J.qstrikes[i] >= 2 ? skip++ : noopt++;
    if (J.qn)
        rt_log("pipe_journal: quarantine %s: %d pipeline(s) built without optimization, %d not built", J.qpath, noopt, skip);
}

int pj_strikes(uint64_t id) {
    if (!J.on) return 0;
    for (int i = 0; i < J.qn; i++)   // filled by pj_open before any compile; read-only afterwards
        if (J.qid[i] == id) return J.qstrikes[i];
    return 0;
}

int pj_begin(uint64_t id, int kind, int strikes) {
    if (!J.on) return -1;
    lock();
    int i = 0;
    while (i < SLOTS && (J.slot[i].flags & IN_USE)) i++;
    if (i < SLOTS) {
        J.slot[i] = (Slot){id, (uint32_t)kind, IN_USE | (strikes == 1 ? NO_OPT : 0)};
        write_slot(i);
    }
    unlock();
    if (i == SLOTS) {
        static int told;
        if (!__atomic_exchange_n(&told, 1, __ATOMIC_RELAXED)) rt_log("pipe_journal: more than %d compiles at once; the extra ones are not recorded", SLOTS);
        return -1;
    }
    my_slot = i;
    return i;
}

void pj_end(int slot) {
    if (slot < 0 || slot >= SLOTS) return;
    lock();
    J.slot[slot] = (Slot){0};
    write_slot(slot);
    unlock();
    if (my_slot == slot) my_slot = -1;
}

void pj_crash(const char *module) {
    if (!J.on) return;
    // Another thread may hold the lock for a write that will never finish (it crashed too, or was stopped with the
    // process): wait a little, then go on without it.
    for (long n = 0; n < 20000000 && __atomic_exchange_n(&J.lock, 1, __ATOMIC_ACQUIRE); n++) __builtin_ia32_pause();
    int mine = my_slot, n = 0;
    for (int i = 0; i < SLOTS; i++) {
        if (!(J.slot[i].flags & IN_USE) || (mine >= 0 ? i != mine : !module)) continue;
        J.slot[i].flags |= BLAMED;
        write_slot(i);
        rt_logf("  while compiling %s pipeline %016llx%s%s; it gets a strike at the next start\n", kind_name(J.slot[i].kind),
                (unsigned long long)J.slot[i].id, J.slot[i].flags & NO_OPT ? " (already without optimization)" : "",
                mine >= 0 ? " on this thread" : " (on another thread)");
        n++;
    }
    if (!n && module) rt_logf("  no pipeline was being compiled\n");
    uint32_t crashed = 1;
    char name[64] = {0};
    if (module) strncpy(name, module, sizeof name - 1);
    write_at((long)offsetof(Header, crashed), &crashed, sizeof crashed);
    write_at((long)offsetof(Header, module), name, sizeof name);
    unlock();
}
