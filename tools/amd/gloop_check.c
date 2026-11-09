// gloop_check: runs runtime/spv_guarded_loop.h on a SPIR-V file, for comparing with tools/amd/spv_guarded.py --looparray.
//   gloop_check <in.spv> <out.spv>     exit 0 rewritten, 1 not rewritten, 2 error
// Build: ferramentas\llvm\bin\clang -O2 -I runtime tools/amd/gloop_check.c -o build/amd-debug/gloop_check.exe
#include <stdio.h>
#include "spv_guarded_loop.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: gloop_check <in.spv> <out.spv>\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)len);
    if (!w || fread(w, 1, (size_t)len, f) != (size_t)len) return 2;
    fclose(f);
    uint64_t n = 0;
    uint32_t b0 = 0, count = 0, set = 0;
    gl_force_table_only = gl_table_only_listed((uint64_t)len / 4);   // OOT_GLOOP_TABLE_ONLY, as the runtime decides
    gl_force_no_pages = gl_no_pages_listed((uint64_t)len / 4);
    gl_force_nostore = gl_nostore_listed((uint64_t)len / 4);
    uint32_t *o = spv_gloop_rewrite(w, (uint64_t)len / 4, &n, &b0, &count, &set);
    if (!o) { printf("%s: not rewritten\n", argv[1]); return 1; }
    f = fopen(argv[2], "wb");
    if (!f || fwrite(o, 4, n, f) != n) return 2;
    fclose(f);
    printf("%s: rewritten, %llu -> %llu words, array of %u at set %u binding %u\n", argv[1], (unsigned long long)len / 4,
           (unsigned long long)n, count, set, b0);
    return 0;
}
