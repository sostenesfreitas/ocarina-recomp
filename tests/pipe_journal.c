// Driver for tests/pipe_journal_test.py: each run is one game process, built with runtime/pipe_journal.c.
//   pipe_journal <mode> <dir> <id hex>
// Modes that end the process call _exit without pj_end, as a crash does; the next run's pj_open reads what they left.
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pipe_journal.h"

void rt_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("[host] ", stdout);
    vprintf(fmt, ap);
    putchar('\n');
    va_end(ap);
}
void rt_logf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

static uint64_t id;

static void *compile_forever(void *arg) {   // a compile on another thread that never ends
    (void)arg;
    pj_begin(id, PJ_GRAPHICS, pj_strikes(id));
    for (;;) sleep(1);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const char *mode = argv[1], *dir = argv[2];
    id = strtoull(argv[3], NULL, 16);
    pj_open(dir, strcmp(mode, "off") != 0);
    printf("strikes %d\n", pj_strikes(id));
    fflush(stdout);
    if (!strcmp(mode, "strikes") || !strcmp(mode, "off")) return 0;
    if (!strcmp(mode, "clean")) {   // a compile that ends, then a crash elsewhere: nothing to blame
        pj_end(pj_begin(id, PJ_GRAPHICS, pj_strikes(id)));
        pj_crash(NULL);
        _exit(3);
    }
    if (!strcmp(mode, "driver")) {   // the driver's compiler faults on this thread while it compiles id
        pj_begin(id, PJ_GRAPHICS, pj_strikes(id));
        pj_crash("nvgpucomp64.dll");
        _exit(3);
    }
    if (!strcmp(mode, "compute")) {
        pj_begin(id, PJ_COMPUTE, pj_strikes(id));
        pj_crash("nvgpucomp64.dll");
        _exit(3);
    }
    if (!strcmp(mode, "own")) {   // a fault in the game's code on a thread that compiles nothing; id compiles elsewhere
        pthread_t t;
        pthread_create(&t, NULL, compile_forever, NULL);
        usleep(200000);
        pj_crash(NULL);
        _exit(3);
    }
    if (!strcmp(mode, "foreign")) {   // a fault in a DLL on a thread that compiles nothing: every compile is blamed
        pthread_t t;
        pthread_create(&t, NULL, compile_forever, NULL);
        usleep(200000);
        pj_crash("nvoglv64.dll");
        _exit(3);
    }
    if (!strcmp(mode, "killed")) {   // closed from the task manager mid-compile: no crash handler ran
        pj_begin(id, PJ_GRAPHICS, pj_strikes(id));
        _exit(1);
    }
    return 2;
}
