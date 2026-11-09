// LOCAL (AMD): guarded storage candidates as a loop. Included by vk_pipeline.c and tools/amd/gloop_check.c; libc only.
//
// The translator reads guest memory it cannot tie to one storage buffer through a helper per operation kind
// (tools/xlate/patches/0015, 0025): `int f(int *lo, int *hi)` tests each candidate buffer in turn and returns the word
// of the first that holds the address: first up to 8 candidates whose base is a constant buffer word, then the 32
// entries of the runtime's table at c0 + 0x800 (vk_exec_draw.c dyn_collect), each candidate a storage buffer of its own.
// A shader calls the helper once per access, up to ~500 times. AMD's compiler (LLPC, drivers 24.7.1 and 26.10.1)
// inlines every call and its time grows faster than the square of the result: 253 calls of a 34-candidate helper never
// finish, nor does a branchless form or DontInline (inlined anyway).
//
// spv_gloop_rewrite turns each such helper into one loop over its candidates j, in their order, so the first match
// still wins: the bases of the constant-buffer candidates are read before the loop, the table's from c0[128 + k] in it,
// and candidate j's buffer is element e(j) of one storage buffer array declared at the first of all the candidates'
// buffers, which must sit on consecutive bindings (the translator numbers them so). The loop index is uniform, so is
// the array index. Every test and access is the one the translator emitted for that candidate; other uses of the
// buffers go through the array with a constant index. The caller merges those bindings into one descriptor array
// (vk_pipeline.c layout_get). Anything that is not exactly this shape is refused (NULL) and left as it is.
// tools/amd/spv_guarded.py --looparray is the first version (table candidates only), kept for experiments.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef GLOOP_C0_LIST   // where the runtime writes the 32 {address, size} entries in c0 (nvn2_c0.h NVN2_C0_LIST)
#define GLOOP_C0_LIST 0x900
#endif
enum { GLOOP_TABLE = 32, GLOOP_VEC0 = GLOOP_C0_LIST / 16, GLOOP_MAX_HELPERS = 64, GLOOP_MAX_STATIC = 128, GLOOP_MAX_BUFFERS = 200 };

typedef struct { uint32_t *p; size_t n, cap; int oom; } GlBuf;

static void gl_put(GlBuf *o, const uint32_t *w, size_t n) {
    if (o->oom || !n) return;
    if (o->n + n > o->cap) {
        size_t cap = o->cap ? o->cap : 4096;
        while (cap < o->n + n) cap *= 2;
        uint32_t *t = realloc(o->p, cap * 4);
        if (!t) { o->oom = 1; return; }
        o->p = t, o->cap = cap;
    }
    memcpy(o->p + o->n, w, n * 4);
    o->n += n;
}

static void gl_ins(GlBuf *o, uint32_t op, int n, const uint32_t *a) {   // an instruction of n operands
    uint32_t h = (uint32_t)(n + 1) << 16 | op;
    gl_put(o, &h, 1);
    gl_put(o, a, (size_t)n);
}
#define GL_INS(o, op, ...) do { const uint32_t a_[] = {__VA_ARGS__}; gl_ins((o), (op), (int)(sizeof a_ / 4), a_); } while (0)

#ifndef GLOOP_PAGES_DEFAULT   // OOT_GLOOP_PAGES unset: vk_pipeline.c makes it vk_caps.amd, as vk_memory.c does
#define GLOOP_PAGES_DEFAULT 0
#endif
#ifndef GLOOP_PT_SHIFT
#define GLOOP_PT_SHIFT 12       // the runtime's page (vk_memory.c PT_SHIFT)
#endif
#ifndef GLOOP_PT_VEC
#define GLOOP_PT_VEC 0xb0       // c0 vec4 holding {table address lo, hi, shift, entries} (nvn2_c0.h NVN2_C0_PAGES / 16)
#endif
enum { GL_UCONVERT = 113, GL_CONVERT_U_TO_PTR = 120, GL_BITCAST = 124, GL_INOTEQUAL = 171, GL_ULT = 176,
       GL_SHR = 194, GL_SHL = 196, GL_OR = 197, GL_AND = 199, GL_LAND = 167, GL_TYPE_FLOAT = 22, GL_EXTENSION = 10,
       GL_EXT_IMPORT = 11, GL_MEMORY_MODEL = 14, GL_CAP_INT64 = 11, GL_CAP_PSB = 5347, GL_AM_PSB64 = 5348,
       GL_SC_PSB = 5349 };
enum { GL_NAME = 5, GL_ENTRY = 15, GL_CAP = 17, GL_TYPE_BOOL = 20, GL_TYPE_INT = 21, GL_TYPE_ARRAY = 28,
       GL_TYPE_POINTER = 32, GL_CONSTANT = 43, GL_FUNCTION = 54, GL_PARAM = 55, GL_FUNCTION_END = 56, GL_CALL = 57,
       GL_VARIABLE = 59, GL_LOAD = 61, GL_STORE = 62, GL_ACCESS = 65, GL_ACCESS_IB = 66, GL_ARRAY_LENGTH = 68,
       GL_DECORATE = 71, GL_MEMBER_DECORATE = 72, GL_IADD = 128, GL_ISUB = 130, GL_SELECT = 169, GL_IEQUAL = 170,
       GL_SLT = 177, GL_ATOMIC_STORE = 228, GL_PHI = 245, GL_LOOP_MERGE = 246, GL_SEL_MERGE = 247, GL_LABEL = 248,
       GL_BRANCH = 249, GL_BRANCH_COND = 250, GL_RETURN = 253, GL_RETURN_VALUE = 254, GL_DEC_BINDING = 33,
       GL_DEC_SET = 34, GL_SC_UNIFORM = 2, GL_SC_FUNCTION = 7, GL_SC_STORAGE = 12, GL_CAP_SSBO_ARRAY_DYNAMIC = 30,
       GL_SGE = 175, GL_BCAST_FIRST = 338, GL_SC_PRIVATE = 6, GL_CAP_GROUP = 61, GL_CAP_BALLOT = 64, GL_SCOPE_SUBGROUP = 3 };

// Instructions a candidate's test and access may hold; anything else and the module stays as it is.
static int gl_allowed(uint32_t op) {
    switch (op) {
    case GL_LOAD: case GL_STORE: case GL_ACCESS: case GL_ARRAY_LENGTH: case 124 /* Bitcast */: case GL_IADD:
    case GL_ISUB: case 132 /* IMul */: case 164: case 165: case 166: case 167: case 168: /* logical */
    case GL_SELECT: case GL_IEQUAL: case 171: case 172: case 174: case 176: case 178: /* integer compares */
    case 194: case 195: case 196: case 197: case 198: case 199: case 200: /* shifts, bitwise */
        return 1;
    default:
        return op >= 227 && op <= 242;   // atomics
    }
}
static int gl_has_result(uint32_t op) { return op != GL_STORE && op != GL_ATOMIC_STORE; }   // among gl_allowed

typedef struct {
    uint32_t test0, test1;              // instruction indices [test0, test1): the test, without label, variables, branch;
                                        // test1 is its OpSelectionMerge, test1 + 1 its OpBranchConditional
    uint32_t then0, then1;              // [then0, then1): the access, without label (then0 - 1) and return (then1)
    uint32_t cond, label;               // the branch condition; the test block's label id
    uint32_t ubo, member, vec, clo, chi;   // its base: ubo[member][vec].clo (low word) and .chi (high word)
    uint32_t ssbo, fvar, elem;          // its storage buffer, its delta variable; the buffer's element in the array
} GlCand;

typedef struct {
    uint32_t f0, f1;                    // OpFunction .. OpFunctionEnd instruction indices
    uint32_t entry_label, entry_vars_end;   // the entry block's label; the first instruction after its variables
    uint32_t ncand, first;              // candidates; the first of the table's (the ones before have constant bases)
    GlCand *c;
    uint32_t final_term;                // the last block's return
} GlHelper;

typedef struct {
    const uint32_t *w;
    uint32_t bound, ninst, *at;         // word offset of each instruction
    uint32_t *def;                      // instruction defining each label, pointer type and variable (~0u none)
    uint32_t *cval;                     // constant value (when cint[id])
    uint8_t *cint;                      // 1: a one-word OpConstant of the i32 type
} GlMod;

static uint32_t gl_op(const GlMod *m, uint32_t i) { return m->w[m->at[i]] & 0xffff; }
static uint32_t gl_len(const GlMod *m, uint32_t i) { return m->w[m->at[i]] >> 16; }
static const uint32_t *gl_a(const GlMod *m, uint32_t i) { return m->w + m->at[i] + 1; }   // operands

// The storage class of the variable id v, or ~0u.
static uint32_t gl_var_class(const GlMod *m, uint32_t v) {
    if (v >= m->bound || m->def[v] == ~0u || gl_op(m, m->def[v]) != GL_VARIABLE || gl_len(m, m->def[v]) < 4) return ~0u;
    return gl_a(m, m->def[v])[2];
}

// The same code in two candidates: same opcodes and operands, results matched in order; each one's own buffer and
// variable, and the base variable, vector and component of its base reads (checked by gl_classify), may differ.
// map: m->bound zero words, zero again on return.
static int gl_same(const GlMod *m, const GlCand *x, const GlCand *y, uint32_t *map) {
    int same = x->test1 - x->test0 == y->test1 - y->test0 && x->then1 - x->then0 == y->then1 - y->then0;
    uint32_t touched[1024], nt = 0;
    for (int part = 0; part < 2 && same; part++) {
        uint32_t i = part ? x->then0 : x->test0, j = part ? y->then0 : y->test0, e = part ? x->then1 : x->test1;
        for (; i < e && same; i++, j++) {
            uint32_t op = gl_op(m, i), len = gl_len(m, i);
            if (op != gl_op(m, j) || len != gl_len(m, j)) { same = 0; break; }
            const uint32_t *a = gl_a(m, i), *b = gl_a(m, j);
            int base_read = op == GL_ACCESS && len == 7 && a[2] == x->ubo;
            if (base_read && (b[2] != y->ubo || a[3] != b[3])) { same = 0; break; }
            for (uint32_t k = 0; k + 1 < len && same; k++) {
                uint32_t u = a[k], v = b[k];
                if (base_read && k >= 2) continue;   // base, vector, component: the candidate's own
                if (gl_has_result(op) && k == 1) {   // x's result stands for y's from here on
                    if (u >= m->bound || nt == 1024) same = 0;
                    else map[u] = v + 1, touched[nt++] = u;
                } else if (u < m->bound && map[u]) {
                    same = map[u] == v + 1;
                } else if (!((u == x->ssbo && v == y->ssbo) || (u == x->fvar && v == y->fvar))) {
                    same = u == v;
                }
            }
        }
    }
    if (same) {   // the branch on the same condition, the same value returned
        same = x->cond < m->bound && map[x->cond] == y->cond + 1;
        if (same && gl_op(m, x->then1) == GL_RETURN_VALUE) {
            uint32_t u = gl_a(m, x->then1)[0], v = gl_a(m, y->then1)[0];
            same = gl_op(m, y->then1) == GL_RETURN_VALUE && u < m->bound && map[u] == v + 1;
        } else if (same) {
            same = gl_op(m, y->then1) == gl_op(m, x->then1);
        }
    }
    for (uint32_t k = 0; k < nt; k++) map[touched[k]] = 0;
    return same;
}

static uint32_t gl_block_end(const GlMod *m, uint32_t label, uint32_t f1) {   // the terminator of the block at label
    uint32_t e = label + 1;
    while (e < f1 && gl_op(m, e) != GL_LABEL) e++;
    return e - 1;
}

// The helper shape of function [f0, f1]: candidates chained by their false branch, each true branch a block that
// returns, the last block a bare return. blk_of: m->bound zero words, zero again on return.
static int gl_shape(const GlMod *m, uint32_t f0, uint32_t f1, uint32_t *blk_of, GlHelper *h) {
    memset(h, 0, sizeof *h);
    h->f0 = f0, h->f1 = f1;
    uint32_t params = 0, labels = 0, cur = 0;
    for (uint32_t i = f0; i < f1; i++) {
        uint32_t op = gl_op(m, i);
        if (op == GL_PARAM) params++;
        if (op == GL_LABEL && gl_len(m, i) == 2 && gl_a(m, i)[0] < m->bound) {
            if (!labels) cur = i;
            blk_of[gl_a(m, i)[0]] = i + 1, labels++;
        }
    }
    int ok = params >= 2 && params <= 4 && labels >= 3 && (h->c = calloc(labels, sizeof *h->c)) != NULL;
    h->entry_label = cur, h->entry_vars_end = cur + 1;
    while (h->entry_vars_end < f1 && gl_op(m, h->entry_vars_end) == GL_VARIABLE) h->entry_vars_end++;
    for (uint32_t steps = 0; ok; steps++) {
        uint32_t t = gl_block_end(m, cur, f1), top = gl_op(m, t);
        if (steps > labels) ok = 0;
        else if (top == GL_RETURN || top == GL_RETURN_VALUE) {   // the last block: a label and a return only
            ok = t == cur + 1 && cur != h->entry_label;
            h->final_term = t;
            break;
        } else if (top != GL_BRANCH_COND || gl_len(m, t) != 4 || t < cur + 2 || gl_op(m, t - 1) != GL_SEL_MERGE ||
                   gl_a(m, t - 1)[0] != gl_a(m, t)[2]) {
            ok = 0;
        } else {
            uint32_t then_id = gl_a(m, t)[1], next_id = gl_a(m, t)[2];
            if (then_id >= m->bound || next_id >= m->bound || !blk_of[then_id] || !blk_of[next_id]) { ok = 0; break; }
            uint32_t tl = blk_of[then_id] - 1, tt = gl_block_end(m, tl, f1);
            if (gl_op(m, tt) != GL_RETURN && gl_op(m, tt) != GL_RETURN_VALUE) { ok = 0; break; }
            GlCand *c = &h->c[h->ncand++];
            c->label = gl_a(m, cur)[0];
            c->test0 = cur == h->entry_label ? h->entry_vars_end : cur + 1, c->test1 = t - 1;
            c->then0 = tl + 1, c->then1 = tt, c->cond = gl_a(m, t)[0];
            if (c->test0 > c->test1) { ok = 0; break; }
            cur = blk_of[next_id] - 1;
        }
    }
    for (uint32_t i = f0; i < f1; i++)
        if (gl_op(m, i) == GL_LABEL && gl_len(m, i) == 2 && gl_a(m, i)[0] < m->bound) blk_of[gl_a(m, i)[0]] = 0;
    if (!ok || h->ncand < GLOOP_TABLE || h->ncand - GLOOP_TABLE > GLOOP_MAX_STATIC) { free(h->c); h->c = NULL; return 0; }
    return 1;
}

// One candidate's parts: its buffer and variable, and its three base reads (ubo[member][vec].clo, .chi, .clo, each an
// access chain its load follows at once). 1 when it has exactly those.
static int gl_parts(const GlMod *m, GlCand *c) {
    uint32_t stores = 0, lengths = 0, reads = 0, comp[3] = {0};
    for (uint32_t i = c->test0; i < c->test1; i++) {
        uint32_t op = gl_op(m, i), len = gl_len(m, i);
        const uint32_t *a = gl_a(m, i);
        if (!gl_allowed(op) || len < 2) return 0;
        if (op == GL_STORE) stores++, c->fvar = a[0];
        if (op == GL_ARRAY_LENGTH) { if (len != 5) return 0; lengths++, c->ssbo = a[2]; }
    }
    if (stores != 1 || lengths != 1) return 0;
    for (uint32_t i = c->test0; i < c->test1; i++) {
        uint32_t op = gl_op(m, i), len = gl_len(m, i);
        const uint32_t *a = gl_a(m, i);
        if (op != GL_ACCESS) continue;
        if (len != 7 || a[2] == c->ssbo || reads == 3 || i + 1 >= c->test1 || gl_op(m, i + 1) != GL_LOAD ||
            gl_len(m, i + 1) != 4 || gl_a(m, i + 1)[2] != a[1])
            return 0;
        if (reads && (a[2] != c->ubo || a[3] != c->member || a[4] != c->vec)) return 0;
        c->ubo = a[2], c->member = a[3], c->vec = a[4], comp[reads++] = a[5];
    }
    if (reads != 3 || comp[2] != comp[0] || comp[1] == comp[0]) return 0;
    c->clo = comp[0], c->chi = comp[1];
    uint32_t accesses = 0;
    for (uint32_t i = c->then0; i < c->then1; i++) {
        uint32_t op = gl_op(m, i);
        if (!gl_allowed(op) || op == GL_ARRAY_LENGTH) return 0;
        if (op == GL_ACCESS && (accesses++, gl_len(m, i) < 4 || gl_a(m, i)[2] != c->ssbo)) return 0;
    }
    uint32_t sc = gl_var_class(m, c->ssbo);
    if (accesses != 1 || (sc != GL_SC_UNIFORM && sc != GL_SC_STORAGE) || gl_var_class(m, c->fvar) != GL_SC_FUNCTION ||
        gl_var_class(m, c->ubo) != GL_SC_UNIFORM)
        return 0;
    for (uint32_t k = 0; k < 4; k++) {   // member, vector and components are constants
        uint32_t id = k == 0 ? c->member : k == 1 ? c->vec : k == 2 ? c->clo : c->chi;
        if (id >= m->bound || !m->cint[id]) return 0;
    }
    if (gl_op(m, c->then1) == GL_RETURN_VALUE && gl_len(m, c->then1) != 2) return 0;
    return 1;
}

// The trailing GLOOP_TABLE candidates are the c0 table's (entry k: c0[128 + k], low word .x, high .y) and every
// candidate has the same code as the table's first.
static int gl_classify(const GlMod *m, GlHelper *h, uint32_t c0var, uint32_t *map) {
    h->first = h->ncand - GLOOP_TABLE;
    for (uint32_t j = 0; j < h->ncand; j++) {
        GlCand *c = &h->c[j];
        if (!gl_parts(m, c)) return 0;
        if (j >= h->first && (c->ubo != c0var || m->cval[c->vec] != GLOOP_VEC0 + (j - h->first) || m->cval[c->clo] != 0 ||
                              m->cval[c->chi] != 1 || c->member != h->c[h->first].member))
            return 0;
    }
    for (uint32_t j = 0; j < h->ncand; j++)
        if (j != h->first && !gl_same(m, &h->c[h->first], &h->c[j], map)) return 0;
    return 1;
}

// Operand positions (from 0) that may hold a pointer, for the instructions that take one.
static int gl_ptr_ops(uint32_t op, uint32_t len, uint32_t *pos, int cap) {
    uint32_t n = len - 1;
    int k = 0;
    if (op == GL_LOAD || op == GL_ACCESS || op == GL_ACCESS_IB || op == 67 || op == 70 || op == GL_ARRAY_LENGTH ||
        op == 83 || op == 60 || (op >= 227 && op <= 242 && op != GL_ATOMIC_STORE)) { if (n > 2) pos[k++] = 2; }
    else if (op == GL_STORE || op == GL_ATOMIC_STORE) { if (n > 0) pos[k++] = 0; }
    else if (op == 63 || op == 64) { if (n > 1) pos[k++] = 0, pos[k++] = 1; }
    else if (op == GL_CALL) { for (uint32_t i = 3; i < n && k < cap; i++) pos[k++] = i; }
    else if (op == GL_PHI || op == GL_SELECT || op == 401 || op == 402) { for (uint32_t i = 2; i < n && k < cap; i++) pos[k++] = i; }
    return k;
}

typedef struct {   // new ids and constants
    uint32_t bound;
    uint32_t cid[256];                  // the i32 constant of value v < 256: the module's last one, or a new one
    uint32_t nnew, newv[256], newid[256];   // constants to declare, in order
} GlIds;

static uint32_t gl_c(GlIds *d, uint32_t v) {   // 0 when v is too large
    if (v >= 256) return 0;
    if (!d->cid[v]) {
        d->cid[v] = d->bound++;
        d->newv[d->nnew] = v, d->newid[d->nnew++] = d->cid[v];
    }
    return d->cid[v];
}

typedef struct {   // gl_candidate's state
    const GlMod *m;
    GlIds *d;
    uint32_t *ren, *touched, nt, bad;
    uint32_t c0var, arr, ptr_t;
    uint32_t hit_dp, hit_ok;   // LOCAL (AMD): OOT_GLOOP_HITADDR: this helper's page pointer for the address and its validity (0: unused)
} GlEmit;

// Candidate c's code (every candidate has the table's first one's) with its base words lo and hi read already and its
// LOCAL (AMD): OOT_GLOOP_HITADDR: the candidate's access block without its return: through the PhysicalStorageBuffer
// pointer dp when dp (loads and stores Aligned 4), else through element e of the array.
static void gl_then_body(GlBuf *b, GlEmit *x, const GlCand *c, uint32_t dp, uint32_t e) {
    const GlMod *m = x->m;
    GlIds *d = x->d;
#define GB_REN(id) ((id) < m->bound && x->ren[id] ? x->ren[id] : (id))
#define GB_RESULT(old, r) do { if ((old) >= m->bound || x->nt == 4096) x->bad = 1; else x->ren[old] = (r), x->touched[x->nt++] = (old); } while (0)
    for (uint32_t i = c->then0; i < c->then1 && !x->bad; i++) {
        uint32_t op = gl_op(m, i), len = gl_len(m, i), t[64];
        const uint32_t *a = gl_a(m, i);
        if (len > 62) { x->bad = 1; break; }
        if (op == GL_ACCESS && a[2] == c->ssbo) {
            if (dp) GB_RESULT(a[1], dp);
            else {
                uint32_t r = d->bound++;
                t[0] = a[0], t[1] = r, t[2] = x->arr, t[3] = e;
                for (uint32_t q = 3; q < len - 1; q++) t[q + 1] = GB_REN(a[q]);
                GB_RESULT(a[1], r);
                gl_ins(b, op, (int)len, t);
            }
        } else if (dp && op == GL_LOAD && GB_REN(a[2]) == dp) {
            uint32_t r = d->bound++;
            GB_RESULT(a[1], r);
            GL_INS(b, GL_LOAD, a[0], r, dp, 2, 4);
        } else if (dp && op == GL_STORE && GB_REN(a[0]) == dp) {
            GL_INS(b, GL_STORE, dp, GB_REN(a[1]), 2, 4);
        } else if (op == GL_ARRAY_LENGTH && a[2] == c->ssbo) {
            if (dp) { x->bad = 1; break; }
            uint32_t ep = d->bound++, r = d->bound++;
            GB_RESULT(a[1], r);
            GL_INS(b, GL_ACCESS, x->ptr_t, ep, x->arr, e);
            GL_INS(b, GL_ARRAY_LENGTH, a[0], r, ep, a[3]);
        } else if (!gl_has_result(op)) {
            for (uint32_t q = 0; q < len - 1; q++) t[q] = GB_REN(a[q]);
            gl_ins(b, op, (int)(len - 1), t);
        } else {
            uint32_t r = d->bound++;
            t[0] = a[0], t[1] = r;
            for (uint32_t q = 2; q < len - 1; q++) t[q] = GB_REN(a[q]);
            GB_RESULT(a[1], r);
            gl_ins(b, op, (int)(len - 1), t);
        }
    }
#undef GB_REN
#undef GB_RESULT
}

// buffer at element e of the array: its test, then a branch to Hit when it holds the address (store_var = store_val
// there, when store_var), its access and the helper's return; to SM when it does not.
static void gl_candidate(GlBuf *b, GlEmit *x, const GlCand *c, uint32_t lo, uint32_t hi, uint32_t e, uint32_t SM,
                         uint32_t Hit, uint32_t store_var, uint32_t store_val) {
    const GlMod *m = x->m;
    GlIds *d = x->d;
    int dual = 0;   // LOCAL (AMD): OOT_GLOOP_HITADDR: the access emitted twice (page address / descriptor), return included
#define GE_REN(id) ((id) < m->bound && x->ren[id] ? x->ren[id] : (id))
#define GE_RESULT(old, r) do { if ((old) >= m->bound || x->nt == 4096) x->bad = 1; else x->ren[old] = (r), x->touched[x->nt++] = (old); } while (0)
    for (int part = 0; part < 2 && !x->bad; part++) {
        if (part) {
            if (c->cond >= m->bound || !x->ren[c->cond]) { x->bad = 1; break; }
            GL_INS(b, GL_SEL_MERGE, SM, 0);
            GL_INS(b, GL_BRANCH_COND, x->ren[c->cond], Hit, SM);
            GL_INS(b, GL_LABEL, Hit);
            if (store_var) GL_INS(b, GL_STORE, store_var, store_val);
            if (x->hit_dp && gl_op(m, c->then1) == GL_RETURN) {   // LOCAL (AMD): OOT_GLOOP_HITADDR: through the page address when the page has one
                uint32_t LP = d->bound++, LD = d->bound++, MX = d->bound++;
                GL_INS(b, GL_SEL_MERGE, MX, 0);
                GL_INS(b, GL_BRANCH_COND, x->hit_ok, LP, LD);
                GL_INS(b, GL_LABEL, LP);
                gl_then_body(b, x, c, x->hit_dp, e);
                GL_INS(b, GL_BRANCH, MX);
                GL_INS(b, GL_LABEL, LD);
                gl_then_body(b, x, c, 0, e);
                GL_INS(b, GL_BRANCH, MX);
                GL_INS(b, GL_LABEL, MX);
                gl_ins(b, GL_RETURN, 0, NULL);
                dual = 1;
                break;
            }
        }
        for (uint32_t i = part ? c->then0 : c->test0; i < (part ? c->then1 : c->test1) && !x->bad; i++) {
            uint32_t op = gl_op(m, i), len = gl_len(m, i), t[64];
            const uint32_t *a = gl_a(m, i);
            if (len > 62) { x->bad = 1; break; }
            if (op == GL_ACCESS && a[2] == x->c0var) {   // a base read: the loaded word is lo or hi
                GE_RESULT(gl_a(m, i + 1)[1], a[5] == c->clo ? lo : hi);
                i++;
            } else if (op == GL_ACCESS && a[2] == c->ssbo) {
                uint32_t r = d->bound++;
                t[0] = a[0], t[1] = r, t[2] = x->arr, t[3] = e;
                for (uint32_t q = 3; q < len - 1; q++) t[q + 1] = GE_REN(a[q]);
                GE_RESULT(a[1], r);
                gl_ins(b, op, (int)len, t);
            } else if (op == GL_ARRAY_LENGTH && a[2] == c->ssbo) {
                uint32_t ep = d->bound++, r = d->bound++;
                GE_RESULT(a[1], r);
                GL_INS(b, GL_ACCESS, x->ptr_t, ep, x->arr, e);
                GL_INS(b, GL_ARRAY_LENGTH, a[0], r, ep, a[3]);
            } else if (!gl_has_result(op)) {
                for (uint32_t q = 0; q < len - 1; q++) t[q] = GE_REN(a[q]);
                gl_ins(b, op, (int)(len - 1), t);
            } else {
                uint32_t r = d->bound++;
                t[0] = a[0], t[1] = r;
                for (uint32_t q = 2; q < len - 1; q++) t[q] = GE_REN(a[q]);
                GE_RESULT(a[1], r);
                gl_ins(b, op, (int)(len - 1), t);
            }
        }
    }
    if (dual) {
    } else if (!x->bad && gl_op(m, c->then1) == GL_RETURN_VALUE) {
        uint32_t v = gl_a(m, c->then1)[0];
        if (v >= m->bound || !x->ren[v]) x->bad = 1;
        else GL_INS(b, GL_RETURN_VALUE, x->ren[v]);
    } else if (!x->bad) {
        gl_ins(b, GL_RETURN, 0, NULL);
    }
    for (uint32_t q = 0; q < x->nt; q++) x->ren[x->touched[q]] = 0;
    x->nt = 0;
#undef GE_REN
#undef GE_RESULT
}

// LOCAL (AMD): OOT_GLOOP_PAGES: the candidate's access block with its storage buffer access chain replaced by the
// PhysicalStorageBuffer pointer dp (loads and stores through it Aligned 4), then its return.
static void gl_then_page(GlBuf *b, GlEmit *x, const GlCand *c, uint32_t dp) {
    const GlMod *m = x->m;
    GlIds *d = x->d;
#define GP_REN(id) ((id) < m->bound && x->ren[id] ? x->ren[id] : (id))
#define GP_RESULT(old, r) do { if ((old) >= m->bound || x->nt == 4096) x->bad = 1; else x->ren[old] = (r), x->touched[x->nt++] = (old); } while (0)
    for (uint32_t i = c->then0; i < c->then1 && !x->bad; i++) {
        uint32_t op = gl_op(m, i), len = gl_len(m, i), t[64];
        const uint32_t *a = gl_a(m, i);
        if (len > 62) { x->bad = 1; break; }
        if (op == GL_ACCESS && a[2] == c->ssbo) {
            GP_RESULT(a[1], dp);
        } else if (op == GL_LOAD && GP_REN(a[2]) == dp) {
            uint32_t r = d->bound++;
            GP_RESULT(a[1], r);
            GL_INS(b, GL_LOAD, a[0], r, dp, 2, 4);
        } else if (op == GL_STORE && GP_REN(a[0]) == dp) {
            GL_INS(b, GL_STORE, dp, GP_REN(a[1]), 2, 4);
        } else if (!gl_has_result(op)) {
            for (uint32_t q = 0; q < len - 1; q++) t[q] = GP_REN(a[q]);
            gl_ins(b, op, (int)(len - 1), t);
        } else {
            uint32_t r = d->bound++;
            t[0] = a[0], t[1] = r;
            for (uint32_t q = 2; q < len - 1; q++) t[q] = GP_REN(a[q]);
            GP_RESULT(a[1], r);
            gl_ins(b, op, (int)(len - 1), t);
        }
    }
    if (!x->bad && gl_op(m, c->then1) == GL_RETURN_VALUE) {
        uint32_t v = gl_a(m, c->then1)[0];
        if (v >= m->bound || !x->ren[v]) x->bad = 1;
        else GL_INS(b, GL_RETURN_VALUE, x->ren[v]);
    } else if (!x->bad) {
        gl_ins(b, GL_RETURN, 0, NULL);
    }
    for (uint32_t q = 0; q < x->nt; q++) x->ren[x->touched[q]] = 0;
    x->nt = 0;
#undef GP_REN
#undef GP_RESULT
}

// The rewritten module (malloc'd, *out_n words) or NULL when it has no helper of this shape or anything about it is
// unexpected. The array of the candidates' buffers: *count buffers at DescriptorSet *set, Binding *first_binding on.
// LOCAL (AMD, debug): OOT_GLOOP_TABLE_ONLY=<source words>[,<source words>] | 1 (every module): the caller sets the flag for a
// module whose source word count is listed; its helpers keep the constant-buffer candidates as the translator chained them,
// first, and only the runtime table becomes the loop (the table_only form). Layout and module must take the same decision.
static _Thread_local int gl_force_table_only;
static _Thread_local int gl_force_no_pages;   // LOCAL (AMD, debug): OOT_GLOOP_NOPAGES=<source words>[,...] | 1: no page table lookup in this module
static int gl_env_words_listed(const char *name, uint64_t src_words, long *list, int *n) {
    if (*n < 0) { *n = 0; const char *e_ = getenv(name); for (char *end; e_ && *e_ && *n < 8; e_ = *end ? end + 1 : end) list[(*n)++] = strtol(e_, &end, 10); }
    for (int i_ = 0; i_ < *n; i_++) if (list[i_] == 1 || list[i_] == (long)src_words) return 1;
    return 0;
}
static _Thread_local int gl_force_nostore;   // LOCAL (AMD): OOT_GLOOP_PAGES_NOSTORE=1 | <source words>,...: this module's stores keep the candidate tests
static int gl_nostore_listed(uint64_t src_words) { static long l[8]; static int n = -1; return gl_env_words_listed("OOT_GLOOP_PAGES_NOSTORE", src_words, l, &n); }
static int gl_no_pages_listed(uint64_t src_words) { static long l[8]; static int n = -1; return gl_env_words_listed("OOT_GLOOP_NOPAGES", src_words, l, &n); }
static int gl_table_only_listed(uint64_t src_words) {
    static long tonly[8]; static int ntonly = -1;
    if (ntonly < 0) { ntonly = 0; const char *e_ = getenv("OOT_GLOOP_TABLE_ONLY"); for (char *end; e_ && *e_ && ntonly < 8; e_ = *end ? end + 1 : end) tonly[ntonly++] = strtol(e_, &end, 10); }
    for (int i_ = 0; i_ < ntonly; i_++) if (tonly[i_] == 1 || tonly[i_] == (long)src_words) return 1;
    return 0;
}

static uint32_t *spv_gloop_rewrite(const uint32_t *w, uint64_t n, uint64_t *out_n, uint32_t *first_binding,
                                   uint32_t *count, uint32_t *set) {
    if (n < 6 || w[0] != 0x07230203 || w[3] > (1u << 22)) return NULL;
    GlMod m = {.w = w, .bound = w[3]};
    for (uint64_t i = 5; i < n; i += w[i] >> 16) {
        if (!(w[i] >> 16) || i + (w[i] >> 16) > n) return NULL;
        m.ninst++;
    }
    uint32_t *res = NULL, *blk_of = calloc(m.bound, 4), *map = calloc(m.bound, 4), *ren = calloc(m.bound, 4);
    uint32_t *binding = malloc((size_t)m.bound * 4), *dset = calloc(m.bound, 4), *kvar = calloc(m.bound, 4);
    uint32_t *use_ep = calloc(m.ninst, 4);   // other uses' ArrayLength: the new access chain's id, per instruction
    GlHelper *hs = calloc(GLOOP_MAX_HELPERS, sizeof *hs);
    GlBuf *body = calloc(GLOOP_MAX_HELPERS, sizeof *body), o = {0};
    GlIds *d = calloc(1, sizeof *d);
    uint32_t nh = 0;
    m.at = malloc((size_t)m.ninst * 4), m.def = malloc((size_t)m.bound * 4), m.cval = calloc(m.bound, 4), m.cint = calloc(m.bound, 1);
    if (!blk_of || !map || !ren || !binding || !dset || !kvar || !use_ep || !hs || !body || !d || !m.at || !m.def ||
        !m.cval || !m.cint)
        goto out;
    memset(m.def, 0xff, (size_t)m.bound * 4);
    memset(binding, 0xff, (size_t)m.bound * 4);
    uint32_t i32 = 0, boolt = 0, c0var = 0, c0names = 0, first_cap = ~0u, first_fn = ~0u, have_cap = 0;
    uint32_t cm1 = 0, ptr_priv = 0, have_group = 0, have_ballot = 0;   // the cache's: -1, Private i32 pointer, capabilities
    uint32_t u32t = 0, u64t = 0, have_i64 = 0, have_psb = 0, first_imp = ~0u, first_mm = ~0u;   // LOCAL (AMD): OOT_GLOOP_PAGES
    {
        uint32_t k = 0;
        for (uint64_t i = 5; i < n; i += w[i] >> 16) m.at[k++] = (uint32_t)i;
    }
    for (uint32_t i = 0; i < m.ninst; i++) {
        uint32_t op = gl_op(&m, i), len = gl_len(&m, i);
        const uint32_t *a = gl_a(&m, i);
        if (op == GL_TYPE_INT && len == 4 && a[1] == 32 && a[2] == 1) i32 = a[0];
        if (op == GL_TYPE_INT && len == 4 && a[1] == 32 && a[2] == 0) u32t = a[0];   // LOCAL (AMD): OOT_GLOOP_PAGES
        if (op == GL_TYPE_INT && len == 4 && a[1] == 64 && a[2] == 0) u64t = a[0];
        if (op == GL_CAP && len == 2 && a[0] == GL_CAP_INT64) have_i64 = 1;
        if (op == GL_CAP && len == 2 && a[0] == GL_CAP_PSB) have_psb = 1;
        if (op == GL_EXT_IMPORT && first_imp == ~0u) first_imp = i;
        if (op == GL_MEMORY_MODEL && first_mm == ~0u) first_mm = i;
        if (op == GL_TYPE_BOOL && len == 2) boolt = a[0];
    }
    for (uint32_t i = 0; i < m.ninst; i++) {
        uint32_t op = gl_op(&m, i), len = gl_len(&m, i);
        const uint32_t *a = gl_a(&m, i);
        if (op == GL_CAP && first_cap == ~0u) first_cap = i;
        if (op == GL_CAP && len == 2 && a[0] == GL_CAP_SSBO_ARRAY_DYNAMIC) have_cap = 1;
        if (op == GL_CAP && len == 2 && a[0] == GL_CAP_GROUP) have_group = 1;
        if (op == GL_CAP && len == 2 && a[0] == GL_CAP_BALLOT) have_ballot = 1;
        if (op == GL_TYPE_POINTER && len == 4 && a[1] == GL_SC_PRIVATE && a[2] == i32 && i32) ptr_priv = a[0];
        if (op == GL_FUNCTION && first_fn == ~0u) first_fn = i;
        if (op == GL_DECORATE && len == 4 && a[0] < m.bound && a[1] == GL_DEC_BINDING) binding[a[0]] = a[2];
        if (op == GL_DECORATE && len == 4 && a[0] < m.bound && a[1] == GL_DEC_SET) dset[a[0]] = a[2];
        if (op == GL_CONSTANT && len == 4 && a[0] == i32 && a[1] < m.bound) {
            m.cval[a[1]] = a[2], m.cint[a[1]] = 1;
            if (a[2] < 256) d->cid[a[2]] = a[1];
            if (a[2] == 0xffffffffu) cm1 = a[1];
        }
        if (op == GL_NAME && len >= 3) {                      // <letters>_c0: the stage's constant buffer 0
            const char *s = (const char *)(a + 1);
            size_t sl = strnlen(s, (size_t)(len - 2) * 4), p = 0;
            while (p < sl && s[p] >= 'a' && s[p] <= 'z') p++;
            if (p && sl == p + 3 && !memcmp(s + p, "_c0", 3)) c0var = a[0], c0names++;
        }
        if ((op == GL_LABEL || op == GL_TYPE_POINTER) && len >= 2 && a[0] < m.bound) m.def[a[0]] = i;   // the ids looked up
        if (op == GL_VARIABLE && len >= 3 && a[1] < m.bound) m.def[a[1]] = i;
    }
    if (!i32 || !boolt || c0names != 1 || first_cap == ~0u || first_fn == ~0u || gl_var_class(&m, c0var) != GL_SC_UNIFORM)
        goto out;
    for (uint32_t i = first_fn; i < m.ninst && nh < GLOOP_MAX_HELPERS; i++) {   // the helpers
        if (gl_op(&m, i) != GL_FUNCTION) continue;
        uint32_t e = i;
        while (e < m.ninst && gl_op(&m, e) != GL_FUNCTION_END) e++;
        if (e == m.ninst) goto out;
        if (gl_shape(&m, i, e, blk_of, &hs[nh])) {
            if (gl_classify(&m, &hs[nh], c0var, map)) nh++;
            else free(hs[nh].c), hs[nh].c = NULL;
        }
        i = e;
    }
    if (!nh) goto out;
    // The array: every buffer on the bindings from the lowest candidate's to the highest's, when each of those is a
    // variable of the candidates' pointer type and set (a buffer between them that is not a candidate joins the array
    // too; its uses take a constant index). Otherwise only the table's 32 buffers, consecutive, and the candidates with
    // constant-buffer bases stay as they are in front of the loop (table_only).
    uint32_t vars[GLOOP_MAX_BUFFERS], nv = 0, ptr_t = 0, table_only = 0, bmin = ~0u, bmax = 0, cset = 0;
    {
        const GlCand *c0c = &hs[0].c[hs[0].first];
        ptr_t = gl_a(&m, m.def[c0c->ssbo])[0], cset = dset[c0c->ssbo];
        for (uint32_t x = 0; x < nh; x++)
            for (uint32_t j = 0; j < hs[x].ncand; j++) {
                uint32_t v = hs[x].c[j].ssbo, bv = binding[v];
                if (bv == ~0u) goto out;
                if (gl_a(&m, m.def[v])[0] != ptr_t || dset[v] != cset) table_only = 1;
                if (bv < bmin) bmin = bv;
                if (bv > bmax) bmax = bv;
            }
        if (!table_only && bmax - bmin + 1 > GLOOP_MAX_BUFFERS) table_only = 1;
        if (gl_force_table_only) table_only = 1;   // LOCAL (AMD, debug): OOT_GLOOP_TABLE_ONLY
        if (!table_only) {   // a variable for each binding of the range
            memset(vars, 0, sizeof vars);
            for (uint32_t i = 0; i < m.ninst && !table_only; i++) {
                if (gl_op(&m, i) != GL_VARIABLE || gl_len(&m, i) < 4) continue;
                uint32_t v = gl_a(&m, i)[1];
                if (v >= m.bound || binding[v] == ~0u || dset[v] != cset || binding[v] < bmin || binding[v] > bmax) continue;
                if (vars[binding[v] - bmin] || gl_a(&m, i)[0] != ptr_t) table_only = 1;
                else vars[binding[v] - bmin] = v;
            }
            nv = bmax - bmin + 1;
            for (uint32_t k = 0; k < nv && !table_only; k++) if (!vars[k]) table_only = 1;
        }
        if (table_only) {   // the table's 32, the same in every helper, consecutive
            nv = GLOOP_TABLE;
            for (uint32_t k = 0; k < GLOOP_TABLE; k++) {
                vars[k] = c0c[k].ssbo;
                for (uint32_t x = 1; x < nh; x++) if (hs[x].c[hs[x].first + k].ssbo != vars[k]) goto out;
                if (gl_a(&m, m.def[vars[k]])[0] != ptr_t || dset[vars[k]] != cset || binding[vars[k]] != binding[vars[0]] + k)
                    goto out;
            }
        }
        for (uint32_t k = 0; k < nv; k++) {
            if (kvar[vars[k]]) goto out;
            kvar[vars[k]] = k + 1;
        }
        for (uint32_t x = 0; x < nh; x++)
            for (uint32_t j = 0; j < hs[x].ncand; j++) {
                GlCand *q = &hs[x].c[j];
                q->elem = kvar[q->ssbo] ? kvar[q->ssbo] - 1 : ~0u;
                if (j >= hs[x].first && (q->elem == ~0u || q->elem != hs[x].c[hs[x].first].elem + (j - hs[x].first))) goto out;
                if (j < hs[x].first && table_only && q->elem != ~0u) goto out;   // kept as it is, it must keep its buffer
            }
    }
    if (ptr_t >= m.bound || m.def[ptr_t] == ~0u || gl_op(&m, m.def[ptr_t]) != GL_TYPE_POINTER) goto out;
    uint32_t sclass = gl_a(&m, m.def[ptr_t])[1], struct_t = gl_a(&m, m.def[ptr_t])[2];
    uint32_t last_dec = ~0u;   // the last decoration that stays: the array's go after it
    for (uint32_t i = 0; i < m.ninst; i++)
        if ((gl_op(&m, i) == GL_DECORATE || gl_op(&m, i) == GL_MEMBER_DECORATE) &&
            !(gl_op(&m, i) == GL_DECORATE && gl_a(&m, i)[0] < m.bound && kvar[gl_a(&m, i)[0]]))
            last_dec = i;
    if (last_dec == ~0u) goto out;

    d->bound = m.bound;
    uint32_t cnv = gl_c(d, nv), arr_t = d->bound++, arr_ptr_t = d->bound++, arr = d->bound++;
    uint32_t cz = gl_c(d, 0), c1 = gl_c(d, 1), cvec0 = gl_c(d, GLOOP_VEC0);
    if (!cnv || !cz || !c1 || !cvec0) goto out;
    // the cache (subgroup operations: SPIR-V 1.3): -1 (none yet), a Private i32 per helper, the subgroup scope
    uint32_t cache = w[1] >= 0x00010300, cscope = gl_c(d, GL_SCOPE_SUBGROUP), new_priv = 0, lastvar[GLOOP_MAX_HELPERS];
    // LOCAL (AMD) measurement: OOT_GLOOP_DIRECT=1: one table entry tested per call (entry 0), no cache, no constant-buffer
    // candidates. Wrong picture: the GPU cost floor of a one-lookup translation (what a page table could reach at best).
    static int gl_direct = -1;
    if (gl_direct < 0) { const char *e_ = getenv("OOT_GLOOP_DIRECT"); gl_direct = e_ && *e_ == '1'; }
    if (gl_direct) cache = 0;
    // LOCAL (AMD): OOT_GLOOP_PAGES: the page table lookup before the loop (gl_then_page; the runtime's table and its c0
    // words: vk_memory.c). New types and constants: unsigned 32 and 64 (when the module has none), the
    // PhysicalStorageBuffer pointers, the 64-bit zero and page mask.
    static int gl_pages = -1;
    if (gl_pages < 0) { const char *e_ = getenv("OOT_GLOOP_PAGES"); gl_pages = e_ && *e_ ? *e_ == '1' : (GLOOP_PAGES_DEFAULT); }
    uint32_t pg_used = 0, pg_u32 = u32t, pg_u64 = u64t, pg_new_u32 = 0, pg_new_u64 = 0, pg_ptr64 = 0, pg_c64z = 0, pg_cmask = 0;
    uint32_t pg_ptrT[8], pg_T[8], pg_nT = 0;
    if (gl_pages) {
        if (!pg_u32) pg_u32 = pg_new_u32 = d->bound++;
        if (!pg_u64) pg_u64 = pg_new_u64 = d->bound++;
        pg_ptr64 = d->bound++, pg_c64z = d->bound++, pg_cmask = d->bound++;
    }
    if (!cscope) goto out;
    if (cache && !cm1) {
        if (d->nnew == 256) goto out;
        cm1 = d->bound++;
        d->newv[d->nnew] = 0xffffffffu, d->newid[d->nnew++] = cm1;
    }
    if (cache && !ptr_priv) ptr_priv = new_priv = d->bound++;
    for (uint32_t x = 0; x < nh; x++) lastvar[x] = cache ? d->bound++ : 0;
    // LOCAL (AMD): OOT_GLOOP_SITECACHE=1: the cached candidate index per CALL SITE of the helper: every call gets a constant
    // site argument (32 slots a helper, further sites share round-robin), the helper a parameter and a new function type, the
    // cache a Private i32[32] indexed by it (a register per site once LLPC inlines the calls). A site stores into one buffer
    // nearly always; the single per-helper entry thrashed when consecutive calls alternate (vegetation setup: 50 ms a dispatch).
    static int gl_sitecache = -1;
    if (gl_sitecache < 0) { const char *e_ = getenv("OOT_GLOOP_SITECACHE"); gl_sitecache = e_ && *e_ == '1'; }
    enum { SC_SLOTS = 32, SC_MAXP = 8 };
    uint32_t sc_on = gl_sitecache && cache, sc_arrT = 0, sc_arrPtrT = 0, sc_init = 0, sc_cslots = 0;
    uint32_t sc_arr[GLOOP_MAX_HELPERS], sc_fnT[GLOOP_MAX_HELPERS], sc_fn[GLOOP_MAX_HELPERS], sc_param[GLOOP_MAX_HELPERS];
    uint32_t sc_sites[GLOOP_MAX_HELPERS], sc_ptypes[GLOOP_MAX_HELPERS][SC_MAXP], sc_np[GLOOP_MAX_HELPERS], sc_ret[GLOOP_MAX_HELPERS];
    memset(sc_sites, 0, sizeof sc_sites);
    if (sc_on) {
        sc_cslots = gl_c(d, SC_SLOTS);
        for (uint32_t s = 0; s < SC_SLOTS; s++) if (!gl_c(d, s)) sc_on = 0;
        if (!sc_cslots) sc_on = 0;
        for (uint32_t x = 0; x < nh && sc_on; x++) {
            sc_fn[x] = gl_a(&m, hs[x].f0)[1], sc_ret[x] = gl_a(&m, hs[x].f0)[0], sc_np[x] = 0;
            for (uint32_t i = hs[x].f0 + 1; i < hs[x].entry_label; i++)
                if (gl_op(&m, i) == GL_PARAM) { if (sc_np[x] == SC_MAXP) sc_on = 0; else sc_ptypes[x][sc_np[x]++] = gl_a(&m, i)[0]; }
        }
    }
    if (sc_on) {
        sc_arrT = d->bound++, sc_arrPtrT = d->bound++, sc_init = d->bound++;
        for (uint32_t x = 0; x < nh; x++) sc_arr[x] = d->bound++, sc_fnT[x] = d->bound++, sc_param[x] = d->bound++;
    }
    // LOCAL (AMD): OOT_GLOOP_MISSCACHE=1: a per-helper window [lo, hi) of addresses no candidate holds, kept per invocation
    // from the last scan that found none (hi = the lowest candidate base above the address). An access inside it drops at
    // once, as the chain would, instead of scanning every candidate again (the vegetation setup shaders fill long runs of
    // such addresses: 50 ms a dispatch with the scan). Only in the full loop form, where both loops see every candidate.
    static int gl_misscache = -1;
    if (gl_misscache < 0) { const char *e_ = getenv("OOT_GLOOP_MISSCACHE"); gl_misscache = e_ && *e_ == '1'; }
    uint32_t mc_on = gl_misscache && gl_pages && cache && !table_only && !gl_direct, mc_ptr = 0, mc_max = 0, mc_used = 0;
    uint32_t mc_lo[GLOOP_MAX_HELPERS], mc_hi[GLOOP_MAX_HELPERS];
    if (mc_on) { mc_ptr = d->bound++; mc_max = d->bound++; for (uint32_t x = 0; x < nh; x++) mc_lo[x] = d->bound++, mc_hi[x] = d->bound++; }
    // Every candidate whose range holds the address reads the same guest word: each buffer is bound at its own base
    // (vk_exec_draw.c guarded_storage_binding binds the real backing, direct_only), so it reads base + (address - base).
    // Which matching candidate answers does not matter, so the helper tries, in this order: the table entry the
    // invocation's previous call matched (per helper, made uniform with OpGroupNonUniformBroadcastFirst, so the array
    // index stays dynamically uniform), the table's 32 entries, then the constant-buffer candidates. Usually the first
    // test hits, where the translator's chain tested up to 90 candidates one after the other.
    uint32_t touched[4096];
    GlEmit E = {.m = &m, .d = d, .ren = ren, .touched = touched, .c0var = c0var, .arr = arr, .ptr_t = ptr_t};
    for (uint32_t x = 0; x < nh; x++) {
        const GlHelper *h = &hs[x];
        const GlCand *c = &h->c[h->first];             // the code every candidate has
        GlBuf *b = &body[x];
        uint32_t S = h->first, ptr_f = 0, f32 = 0;
        uint32_t cS = gl_c(d, S), cE0 = gl_c(d, c->elem), c32 = gl_c(d, GLOOP_TABLE);
        for (uint32_t i = c->test0; i < c->test1; i++)   // the base reads' types, from the table's first candidate
            if (gl_op(&m, i) == GL_ACCESS && gl_a(&m, i)[2] == c0var) { ptr_f = gl_a(&m, i)[0], f32 = gl_a(&m, i + 1)[0]; break; }
        if (!cS || !cE0 || !c32 || !ptr_f) goto out;
        for (uint32_t i = h->f0; i <= h->entry_label; i++) {   // OpFunction, its parameters, the entry label
            if (sc_on && i == h->f0) {   // LOCAL (AMD): OOT_GLOOP_SITECACHE: the function with its new type
                GL_INS(b, GL_FUNCTION, gl_a(&m, i)[0], gl_a(&m, i)[1], gl_a(&m, i)[2], sc_fnT[x]);
                continue;
            }
            if (sc_on && i == h->entry_label) GL_INS(b, GL_PARAM, i32, sc_param[x]);   // the site argument, last
            if (gl_op(&m, i) != GL_LABEL || i == h->entry_label) gl_put(b, w + m.at[i], gl_len(&m, i));
        }
        for (uint32_t i = h->entry_label + 1; i < h->entry_vars_end; i++) gl_put(b, w + m.at[i], gl_len(&m, i));
        uint32_t pred = gl_a(&m, h->entry_label)[0];
        uint32_t sc_slot = 0;   // LOCAL (AMD): OOT_GLOOP_SITECACHE: this site's slot of the cached index
        if (sc_on) { sc_slot = d->bound++; GL_INS(b, GL_ACCESS, ptr_priv, sc_slot, sc_arr[x], sc_param[x]); }
        if (table_only) {                              // table_only: the other candidates as they are, first
            for (uint32_t s = 0; s < S; s++) {
                const GlCand *q = &h->c[s];
                for (uint32_t i = s ? m.def[q->label] : q->test0; i <= q->test1 + 1; i++) gl_put(b, w + m.at[i], gl_len(&m, i));
                for (uint32_t i = q->then0 - 1; i <= q->then1; i++) gl_put(b, w + m.at[i], gl_len(&m, i));
            }
            if (S) {                                   // the last one's false branch
                pred = c->label;
                GL_INS(b, GL_LABEL, pred);
            }
        }
#define GL_TABLE_READ(kid, lo, hi, e) do { \
            uint32_t v_ = d->bound++, pl_ = d->bound++, ph_ = d->bound++; \
            lo = d->bound++, hi = d->bound++, e = d->bound++; \
            GL_INS(b, GL_IADD, i32, v_, kid, cvec0); \
            GL_INS(b, GL_ACCESS, ptr_f, pl_, c0var, c->member, v_, c->clo); \
            GL_INS(b, GL_LOAD, f32, lo, pl_); \
            GL_INS(b, GL_ACCESS, ptr_f, ph_, c0var, c->member, v_, c->chi); \
            GL_INS(b, GL_LOAD, f32, hi, ph_); \
            GL_INS(b, GL_IADD, i32, e, kid, cE0); } while (0)
        uint32_t mc_x = 0, mc_va = 0, mc_aok = 0, mc_nb = 0, mc_k32 = 0;   // LOCAL (AMD): OOT_GLOOP_MISSCACHE, this helper
        if (mc_on) {
            uint32_t q0 = 0, q1 = 0;
            if (h->f0 + 2 < h->f1 && gl_op(&m, h->f0 + 1) == GL_PARAM && gl_op(&m, h->f0 + 2) == GL_PARAM)
                q0 = gl_a(&m, h->f0 + 1)[1], q1 = gl_a(&m, h->f0 + 2)[1];
            uint32_t k3 = gl_c(d, 3);
            mc_k32 = gl_c(d, 32);
            mc_x = q0 && q1 && k3 && mc_k32 && gl_op(&m, c->test0) == GL_LOAD && gl_a(&m, c->test0)[2] == q0;
            if (mc_x) {
                uint32_t l_ = d->bound++, h_ = d->bound++, lu = d->bound++, hu = d->bound++, l64 = d->bound++, h64 = d->bound++;
                uint32_t hs_ = d->bound++, ml = d->bound++, mh = d->bound++, ge = d->bound++, lt = d->bound++, al = d->bound++;
                uint32_t hit = d->bound++, hit2 = d->bound++, D = d->bound++, M = d->bound++;
                mc_va = d->bound++, mc_aok = d->bound++;
                GL_INS(b, GL_LOAD, i32, l_, q0);
                GL_INS(b, GL_LOAD, i32, h_, q1);
                GL_INS(b, GL_BITCAST, pg_u32, lu, l_);
                GL_INS(b, GL_BITCAST, pg_u32, hu, h_);
                GL_INS(b, GL_UCONVERT, pg_u64, l64, lu);
                GL_INS(b, GL_UCONVERT, pg_u64, h64, hu);
                GL_INS(b, GL_SHL, pg_u64, hs_, h64, mc_k32);
                GL_INS(b, GL_OR, pg_u64, mc_va, hs_, l64);
                GL_INS(b, GL_LOAD, pg_u64, ml, mc_lo[x]);
                GL_INS(b, GL_LOAD, pg_u64, mh, mc_hi[x]);
                GL_INS(b, 174 /* UGreaterThanEqual */, boolt, ge, mc_va, ml);
                GL_INS(b, GL_ULT, boolt, lt, mc_va, mh);
                GL_INS(b, GL_AND, i32, al, l_, k3);
                GL_INS(b, GL_IEQUAL, boolt, mc_aok, al, cz);
                GL_INS(b, GL_LAND, boolt, hit, ge, lt);
                GL_INS(b, GL_LAND, boolt, hit2, hit, mc_aok);
                GL_INS(b, GL_SEL_MERGE, M, 0);
                GL_INS(b, GL_BRANCH_COND, hit2, D, M);
                GL_INS(b, GL_LABEL, D);
                gl_put(b, w + m.at[h->final_term], gl_len(&m, h->final_term));   // inside the window: the chain's answer
                GL_INS(b, GL_LABEL, M);
                pred = M;
                mc_used = 1;
                pg_used = 1;
            }
        }
        // the running bound: the lowest candidate base above the address (lo, hi: the base as the c0 floats)
#define GL_MC_BOUND(lo_, hi_, nb_, nbn_) do { \
            uint32_t lu_ = d->bound++, hu_ = d->bound++, l64_ = d->bound++, h64_ = d->bound++, hs2_ = d->bound++; \
            uint32_t b64_ = d->bound++, ab_ = d->bound++, lt_ = d->bound++, bo_ = d->bound++; \
            GL_INS(b, GL_BITCAST, pg_u32, lu_, lo_); \
            GL_INS(b, GL_BITCAST, pg_u32, hu_, hi_); \
            GL_INS(b, GL_UCONVERT, pg_u64, l64_, lu_); \
            GL_INS(b, GL_UCONVERT, pg_u64, h64_, hu_); \
            GL_INS(b, GL_SHL, pg_u64, hs2_, h64_, mc_k32); \
            GL_INS(b, GL_OR, pg_u64, b64_, hs2_, l64_); \
            GL_INS(b, 172 /* UGreaterThan */, boolt, ab_, b64_, mc_va); \
            GL_INS(b, GL_ULT, boolt, lt_, b64_, nb_); \
            GL_INS(b, GL_LAND, boolt, bo_, ab_, lt_); \
            GL_INS(b, GL_SELECT, pg_u64, nbn_, bo_, b64_, nb_); } while (0)
        E.hit_dp = E.hit_ok = 0;   // LOCAL (AMD): OOT_GLOOP_HITADDR
        if (gl_pages && ptr_f && f32 && !gl_force_no_pages) {   // LOCAL (AMD): OOT_GLOOP_PAGES: 0. one page table lookup (not with OOT_GLOOP_NOPAGES)
            uint32_t p0 = 0, p1 = 0, T = 0, acc = 0, nacc = 0;
            if (h->f0 + 2 < h->f1 && gl_op(&m, h->f0 + 1) == GL_PARAM && gl_op(&m, h->f0 + 2) == GL_PARAM)
                p0 = gl_a(&m, h->f0 + 1)[1], p1 = gl_a(&m, h->f0 + 2)[1];
            for (uint32_t i = c->then0; i < c->then1; i++)
                if (gl_op(&m, i) == GL_ACCESS && gl_a(&m, i)[2] == c->ssbo) acc = i, nacc++;
            int plo_first = p0 && gl_op(&m, c->test0) == GL_LOAD && gl_a(&m, c->test0)[2] == p0;
            int pg_store = 0;   // LOCAL (AMD): OOT_GLOOP_PAGES_NOSTORE=1: a store helper keeps its candidate tests (the chain drops a store
            int gl_pg_nostore = gl_force_nostore;   // whose address no bound buffer holds; the page path stored to any mapped address)
            static int gl_hitaddr = -1;   // LOCAL (AMD): OOT_GLOOP_HITADDR=1: a store helper writes through the page address once a candidate matched
            if (gl_hitaddr < 0) { const char *e_ = getenv("OOT_GLOOP_HITADDR"); gl_hitaddr = e_ && *e_ == '1'; }
            if (nacc == 1) for (uint32_t i = c->then0; i < c->then1; i++)
                if (gl_op(&m, i) == GL_STORE && gl_a(&m, i)[0] == gl_a(&m, acc)[1]) pg_store = 1;
            if (nacc == 1 && plo_first && (!(pg_store && gl_pg_nostore) || gl_hitaddr)) {   // a store helper under NOSTORE allocates nothing unless HITADDR (the module stays the precompiled one)
                uint32_t pt = gl_a(&m, acc)[0];
                if (pt < m.bound && m.def[pt] != ~0u && gl_op(&m, m.def[pt]) == GL_TYPE_POINTER) T = gl_a(&m, m.def[pt])[2];
            }
            uint32_t tdef = ~0u;   // the element type's declaration (m.def holds labels, pointer types and variables only)
            for (uint32_t i = 0; T && i < first_fn; i++)
                if ((gl_op(&m, i) == GL_TYPE_INT || gl_op(&m, i) == GL_TYPE_FLOAT) && gl_len(&m, i) >= 3 && gl_a(&m, i)[0] == T) tdef = i;
            int t32 = tdef != ~0u && gl_len(&m, tdef) >= 3 &&
                      ((gl_op(&m, tdef) == GL_TYPE_INT && gl_a(&m, tdef)[1] == 32) || (gl_op(&m, tdef) == GL_TYPE_FLOAT && gl_a(&m, tdef)[1] == 32));
            uint32_t ptrT = 0;
            for (uint32_t q = 0; q < pg_nT; q++) if (pg_T[q] == T) ptrT = pg_ptrT[q];
            if (t32 && !ptrT && pg_nT < 8) pg_T[pg_nT] = T, ptrT = pg_ptrT[pg_nT++] = d->bound++;
            uint32_t k32 = gl_c(d, 32), k3 = gl_c(d, 3), ksh = gl_c(d, GLOOP_PT_SHIFT), kpt = gl_c(d, GLOOP_PT_VEC), kw = gl_c(d, 3);
#ifdef GLOOP_DEBUG
            fprintf(stderr, "gloop pages: helper %u: p0 %u p1 %u nacc %u plo_first %d T %u t32 %d ptrT %u consts %u %u %u %u test0 op %u ptr %u\n",
                    x, p0, p1, nacc, plo_first, T, t32, ptrT, k32, k3, ksh, kpt, gl_op(&m, c->test0), gl_a(&m, c->test0)[2]);
            fprintf(stderr, "gloop pages: tdef %u op %u len %u a1 %u a2 %u\n", tdef, tdef != ~0u ? gl_op(&m, tdef) : 0,
                    tdef != ~0u ? gl_len(&m, tdef) : 0, tdef != ~0u ? gl_a(&m, tdef)[1] : 0, tdef != ~0u ? gl_a(&m, tdef)[2] : 0);
#endif
            if (t32 && ptrT && k32 && k3 && ksh && kpt && kw && !(pg_store && gl_pg_nostore)) {
                uint32_t lo_ = d->bound++, hi_ = d->bound++, lou = d->bound++, hiu = d->bound++, lo64 = d->bound++, hi64 = d->bound++;
                uint32_t hs = d->bound++, va = d->bound++, tx = d->bound++, ty = d->bound++, tw = d->bound++;
                uint32_t tx64 = d->bound++, ty64 = d->bound++, tys = d->bound++, tbl = d->bound++, tn64 = d->bound++;
                uint32_t idx = d->bound++, okt = d->bound++, oki = d->bound++, al = d->bound++, oka = d->bound++, ok1 = d->bound++, ok = d->bound++;
                uint32_t M0 = d->bound++, L1 = d->bound++, e8 = d->bound++, ea = d->bound++, ep = d->bound++, en = d->bound++;
                uint32_t ok2 = d->bound++, M1 = d->bound++, L2 = d->bound++, off = d->bound++, da = d->bound++, dp = d->bound++;
                GL_INS(b, GL_LOAD, i32, lo_, p0);
                GL_INS(b, GL_LOAD, i32, hi_, p1);
                GL_INS(b, GL_BITCAST, pg_u32, lou, lo_);
                GL_INS(b, GL_BITCAST, pg_u32, hiu, hi_);
                GL_INS(b, GL_UCONVERT, pg_u64, lo64, lou);
                GL_INS(b, GL_UCONVERT, pg_u64, hi64, hiu);
                GL_INS(b, GL_SHL, pg_u64, hs, hi64, k32);
                GL_INS(b, GL_OR, pg_u64, va, hs, lo64);
                uint32_t comp[3] = {cz, c1, kw}, out3[3] = {tx, ty, tw};
                for (int q = 0; q < 3; q++) {
                    uint32_t pa = d->bound++, fv = d->bound++;
                    GL_INS(b, GL_ACCESS, ptr_f, pa, c0var, c->member, kpt, comp[q]);
                    GL_INS(b, GL_LOAD, f32, fv, pa);
                    GL_INS(b, GL_BITCAST, pg_u32, out3[q], fv);
                }
                GL_INS(b, GL_UCONVERT, pg_u64, tx64, tx);
                GL_INS(b, GL_UCONVERT, pg_u64, ty64, ty);
                GL_INS(b, GL_SHL, pg_u64, tys, ty64, k32);
                GL_INS(b, GL_OR, pg_u64, tbl, tys, tx64);
                GL_INS(b, GL_UCONVERT, pg_u64, tn64, tw);
                GL_INS(b, GL_SHR, pg_u64, idx, va, ksh);
                GL_INS(b, GL_INOTEQUAL, boolt, okt, tbl, pg_c64z);
                GL_INS(b, GL_ULT, boolt, oki, idx, tn64);
                GL_INS(b, GL_AND, i32, al, lo_, k3);
                GL_INS(b, GL_IEQUAL, boolt, oka, al, cz);
                GL_INS(b, GL_LAND, boolt, ok1, okt, oki);
                GL_INS(b, GL_LAND, boolt, ok, ok1, oka);
                GL_INS(b, GL_SEL_MERGE, M0, 0);
                GL_INS(b, GL_BRANCH_COND, ok, L1, M0);
                GL_INS(b, GL_LABEL, L1);
                GL_INS(b, GL_SHL, pg_u64, e8, idx, k3);
                GL_INS(b, GL_IADD, pg_u64, ea, tbl, e8);
                GL_INS(b, GL_CONVERT_U_TO_PTR, pg_ptr64, ep, ea);
                GL_INS(b, GL_LOAD, pg_u64, en, ep, 2, 8);
                GL_INS(b, GL_INOTEQUAL, boolt, ok2, en, pg_c64z);
                GL_INS(b, GL_SEL_MERGE, M1, 0);
                GL_INS(b, GL_BRANCH_COND, ok2, L2, M1);
                GL_INS(b, GL_LABEL, L2);
                GL_INS(b, GL_AND, pg_u64, off, va, pg_cmask);
                GL_INS(b, GL_IADD, pg_u64, da, en, off);
                GL_INS(b, GL_CONVERT_U_TO_PTR, ptrT, dp, da);
                gl_then_page(b, &E, c, dp);
                GL_INS(b, GL_LABEL, M1);
                GL_INS(b, GL_BRANCH, M0);
                GL_INS(b, GL_LABEL, M0);
                pred = M0;
                pg_used = 1;
            } else if (t32 && ptrT && k32 && k3 && ksh && kpt && kw && pg_store && gl_pg_nostore && gl_hitaddr) {
                // LOCAL (AMD): OOT_GLOOP_HITADDR: the page address of the store's target, for the candidate that will match (phis: 0 when none)
                uint32_t lo_ = d->bound++, hi_ = d->bound++, lou = d->bound++, hiu = d->bound++, lo64 = d->bound++, hi64 = d->bound++;
                uint32_t hs = d->bound++, va = d->bound++, tx = d->bound++, ty = d->bound++, tw = d->bound++;
                uint32_t tx64 = d->bound++, ty64 = d->bound++, tys = d->bound++, tbl = d->bound++, tn64 = d->bound++;
                uint32_t idx = d->bound++, okt = d->bound++, oki = d->bound++, al = d->bound++, oka = d->bound++, ok1 = d->bound++, ok = d->bound++;
                uint32_t M0 = d->bound++, L1 = d->bound++, e8 = d->bound++, ea = d->bound++, ep = d->bound++, en = d->bound++;
                uint32_t off = d->bound++, da = d->bound++, enp = d->bound++, dap = d->bound++, okd = d->bound++, dp = d->bound++;
                GL_INS(b, GL_LOAD, i32, lo_, p0);
                GL_INS(b, GL_LOAD, i32, hi_, p1);
                GL_INS(b, GL_BITCAST, pg_u32, lou, lo_);
                GL_INS(b, GL_BITCAST, pg_u32, hiu, hi_);
                GL_INS(b, GL_UCONVERT, pg_u64, lo64, lou);
                GL_INS(b, GL_UCONVERT, pg_u64, hi64, hiu);
                GL_INS(b, GL_SHL, pg_u64, hs, hi64, k32);
                GL_INS(b, GL_OR, pg_u64, va, hs, lo64);
                {
                    uint32_t comp[3] = {cz, c1, kw}, out3[3] = {tx, ty, tw};
                    for (int q = 0; q < 3; q++) {
                        uint32_t pa = d->bound++, fv = d->bound++;
                        GL_INS(b, GL_ACCESS, ptr_f, pa, c0var, c->member, kpt, comp[q]);
                        GL_INS(b, GL_LOAD, f32, fv, pa);
                        GL_INS(b, GL_BITCAST, pg_u32, out3[q], fv);
                    }
                }
                GL_INS(b, GL_UCONVERT, pg_u64, tx64, tx);
                GL_INS(b, GL_UCONVERT, pg_u64, ty64, ty);
                GL_INS(b, GL_SHL, pg_u64, tys, ty64, k32);
                GL_INS(b, GL_OR, pg_u64, tbl, tys, tx64);
                GL_INS(b, GL_UCONVERT, pg_u64, tn64, tw);
                GL_INS(b, GL_SHR, pg_u64, idx, va, ksh);
                GL_INS(b, GL_INOTEQUAL, boolt, okt, tbl, pg_c64z);
                GL_INS(b, GL_ULT, boolt, oki, idx, tn64);
                GL_INS(b, GL_AND, i32, al, lo_, k3);
                GL_INS(b, GL_IEQUAL, boolt, oka, al, cz);
                GL_INS(b, GL_LAND, boolt, ok1, okt, oki);
                GL_INS(b, GL_LAND, boolt, ok, ok1, oka);
                GL_INS(b, GL_SEL_MERGE, M0, 0);
                GL_INS(b, GL_BRANCH_COND, ok, L1, M0);
                GL_INS(b, GL_LABEL, L1);
                GL_INS(b, GL_SHL, pg_u64, e8, idx, k3);
                GL_INS(b, GL_IADD, pg_u64, ea, tbl, e8);
                GL_INS(b, GL_CONVERT_U_TO_PTR, pg_ptr64, ep, ea);
                GL_INS(b, GL_LOAD, pg_u64, en, ep, 2, 8);
                GL_INS(b, GL_AND, pg_u64, off, va, pg_cmask);
                GL_INS(b, GL_IADD, pg_u64, da, en, off);
                GL_INS(b, GL_BRANCH, M0);
                GL_INS(b, GL_LABEL, M0);
                GL_INS(b, GL_PHI, pg_u64, enp, en, L1, pg_c64z, pred);
                GL_INS(b, GL_PHI, pg_u64, dap, da, L1, pg_c64z, pred);
                GL_INS(b, GL_INOTEQUAL, boolt, okd, enp, pg_c64z);
                GL_INS(b, GL_CONVERT_U_TO_PTR, ptrT, dp, dap);
                pred = M0;
                E.hit_dp = dp, E.hit_ok = okd;
                pg_used = 1;
            }
        }
        uint32_t lo, hi, e;
        if (cache) {                                   // 1. the entry the previous call matched
            uint32_t lk = d->bound++, kc = d->bound++, ok = d->bound++, C1 = d->bound++, H1 = d->bound++, S1 = d->bound++,
                     M1 = d->bound++;
            GL_INS(b, GL_LOAD, i32, lk, sc_on ? sc_slot : lastvar[x]);
            GL_INS(b, GL_BCAST_FIRST, i32, kc, cscope, lk);
            GL_INS(b, GL_SGE, boolt, ok, kc, cz);
            GL_INS(b, GL_SEL_MERGE, M1, 0);
            GL_INS(b, GL_BRANCH_COND, ok, C1, M1);
            GL_INS(b, GL_LABEL, C1);
            GL_TABLE_READ(kc, lo, hi, e);
            gl_candidate(b, &E, c, lo, hi, e, S1, H1, 0, 0);
            GL_INS(b, GL_LABEL, S1);
            GL_INS(b, GL_BRANCH, M1);
            GL_INS(b, GL_LABEL, M1);
            pred = M1;
        }
        {                                              // 2. the table's entries
            uint32_t HA = d->bound++, BA = d->bound++, HitA = d->bound++, SA = d->bound++, CA = d->bound++, XA = d->bound++;
            uint32_t k = d->bound++, k1 = d->bound++, more = d->bound++;
            GL_INS(b, GL_BRANCH, HA);
            GL_INS(b, GL_LABEL, HA);
            GL_INS(b, GL_PHI, i32, k, cz, pred, k1, CA);
            uint32_t nbA = 0, nbA_next = 0;   // LOCAL (AMD): OOT_GLOOP_MISSCACHE
            if (mc_x) { nbA = d->bound++; nbA_next = d->bound++; GL_INS(b, GL_PHI, pg_u64, nbA, mc_max, pred, nbA_next, CA); }
            GL_INS(b, GL_LOOP_MERGE, XA, CA, 2);      // DontUnroll
            GL_INS(b, GL_BRANCH, BA);
            GL_INS(b, GL_LABEL, BA);
            GL_TABLE_READ(k, lo, hi, e);
            if (mc_x) GL_MC_BOUND(lo, hi, nbA, nbA_next);
            gl_candidate(b, &E, c, lo, hi, e, SA, HitA, cache ? (sc_on ? sc_slot : lastvar[x]) : 0, k);
            GL_INS(b, GL_LABEL, SA);
            GL_INS(b, GL_BRANCH, CA);
            GL_INS(b, GL_LABEL, CA);
            GL_INS(b, GL_IADD, i32, k1, k, c1);
            GL_INS(b, GL_SLT, boolt, more, k1, gl_direct ? c1 : c32);   // LOCAL (AMD): OOT_GLOOP_DIRECT
            GL_INS(b, GL_BRANCH_COND, more, HA, XA);
            GL_INS(b, GL_LABEL, XA);
            pred = XA;
            if (mc_x) mc_nb = nbA_next;
        }
#undef GL_TABLE_READ
        if (!table_only && S && !gl_direct) {          // 3. the constant-buffer candidates (LOCAL (AMD): none with OOT_GLOOP_DIRECT)
            uint32_t lo_s[GLOOP_MAX_STATIC], hi_s[GLOOP_MAX_STATIC];
            for (uint32_t s = 0; s < S; s++) {         // their bases, read here only
                const GlCand *q = &h->c[s];
                uint32_t pl = d->bound++, l = d->bound++, ph = d->bound++, hh = d->bound++;
                GL_INS(b, GL_ACCESS, ptr_f, pl, q->ubo, q->member, q->vec, q->clo);
                GL_INS(b, GL_LOAD, f32, l, pl);
                GL_INS(b, GL_ACCESS, ptr_f, ph, q->ubo, q->member, q->vec, q->chi);
                GL_INS(b, GL_LOAD, f32, hh, ph);
                lo_s[s] = l, hi_s[s] = hh;
            }
            uint32_t HB = d->bound++, BB = d->bound++, HitB = d->bound++, SB = d->bound++, CB = d->bound++, XB = d->bound++;
            uint32_t j = d->bound++, j1 = d->bound++, more = d->bound++;
            GL_INS(b, GL_BRANCH, HB);
            GL_INS(b, GL_LABEL, HB);
            GL_INS(b, GL_PHI, i32, j, cz, pred, j1, CB);
            uint32_t nbB = 0, nbB_next = 0;   // LOCAL (AMD): OOT_GLOOP_MISSCACHE
            if (mc_x) { nbB = d->bound++; nbB_next = d->bound++; GL_INS(b, GL_PHI, pg_u64, nbB, mc_nb, pred, nbB_next, CB); }
            GL_INS(b, GL_LOOP_MERGE, XB, CB, 2);
            GL_INS(b, GL_BRANCH, BB);
            GL_INS(b, GL_LABEL, BB);
            uint32_t run = 1;   // their buffers follow in their order: e = j + the first's element
            for (uint32_t s = 1; s < S && run; s++) run = h->c[s].elem == h->c[0].elem + s;
            lo = lo_s[S - 1], hi = hi_s[S - 1], e = gl_c(d, h->c[S - 1].elem);
            if (!e) goto out;
            if (run) {
                uint32_t coff = gl_c(d, h->c[0].elem), e2 = d->bound++;
                if (!coff) goto out;
                GL_INS(b, GL_IADD, i32, e2, j, coff);
                e = e2;
            }
            for (uint32_t s = S - 1; s-- > 0;) {
                uint32_t cs = gl_c(d, s), eq = d->bound++, l2 = d->bound++, h2 = d->bound++;
                if (!cs) goto out;
                GL_INS(b, GL_IEQUAL, boolt, eq, j, cs);
                GL_INS(b, GL_SELECT, f32, l2, eq, lo_s[s], lo);
                GL_INS(b, GL_SELECT, f32, h2, eq, hi_s[s], hi);
                if (!run) {
                    uint32_t ce = gl_c(d, h->c[s].elem), e2 = d->bound++;
                    if (!ce) goto out;
                    GL_INS(b, GL_SELECT, i32, e2, eq, ce, e);
                    e = e2;
                }
                lo = l2, hi = h2;
            }
            if (mc_x) GL_MC_BOUND(lo, hi, nbB, nbB_next);
            gl_candidate(b, &E, c, lo, hi, e, SB, HitB, 0, 0);
            GL_INS(b, GL_LABEL, SB);
            GL_INS(b, GL_BRANCH, CB);
            GL_INS(b, GL_LABEL, CB);
            GL_INS(b, GL_IADD, i32, j1, j, c1);
            GL_INS(b, GL_SLT, boolt, more, j1, cS);
            GL_INS(b, GL_BRANCH_COND, more, HB, XB);
            GL_INS(b, GL_LABEL, XB);
            if (mc_x) mc_nb = nbB_next;
        }
#undef GL_MC_BOUND
        if (mc_x) {   // LOCAL (AMD): OOT_GLOOP_MISSCACHE: none matched: remember [address, next base) (aligned addresses only)
            uint32_t slo = d->bound++, shi = d->bound++;
            GL_INS(b, GL_SELECT, pg_u64, slo, mc_aok, mc_va, mc_max);
            GL_INS(b, GL_SELECT, pg_u64, shi, mc_aok, mc_nb, pg_c64z);
            GL_INS(b, GL_STORE, mc_lo[x], slo);
            GL_INS(b, GL_STORE, mc_hi[x], shi);
        }
        gl_put(b, w + m.at[h->final_term], gl_len(&m, h->final_term));   // none matched
        gl_ins(b, GL_FUNCTION_END, 0, NULL);
        if (b->oom || E.bad) goto out;
    }
    // the other uses of the buffers, in module order: ids and constants first
    for (uint32_t i = 0, x = 0; i < m.ninst; i++) {
        if (x < nh && i == hs[x].f0) { i = hs[x].f1, x++; continue; }
        uint32_t op = gl_op(&m, i), len = gl_len(&m, i), pos[64];
        const uint32_t *a = gl_a(&m, i);
        if ((op == GL_ACCESS || op == GL_ACCESS_IB) && len >= 4 && a[2] < m.bound && kvar[a[2]]) {
            if (!gl_c(d, kvar[a[2]] - 1) || len > 60) goto out;
        } else if (op == GL_ARRAY_LENGTH && len == 5 && a[2] < m.bound && kvar[a[2]]) {
            use_ep[i] = d->bound++;
            if (!gl_c(d, kvar[a[2]] - 1)) goto out;
        } else if (!((op == GL_VARIABLE && len >= 3 && a[1] < m.bound && kvar[a[1]]) ||
                     ((op == GL_DECORATE || op == GL_NAME) && len >= 2 && a[0] < m.bound && kvar[a[0]]) || op == GL_ENTRY)) {
            int np = gl_ptr_ops(op, len, pos, 64);
            for (int q = 0; q < np; q++) if (a[pos[q]] < m.bound && kvar[a[pos[q]]]) goto out;
        }
    }
    // the module
    gl_put(&o, w, 5);
    for (uint32_t i = 0, x = 0; i < m.ninst; i++) {
        uint32_t op = gl_op(&m, i), len = gl_len(&m, i), t[260];
        const uint32_t *a = gl_a(&m, i);
        if (i == first_cap && !have_cap) GL_INS(&o, GL_CAP, GL_CAP_SSBO_ARRAY_DYNAMIC);
        if (i == first_cap && cache && !have_group) GL_INS(&o, GL_CAP, GL_CAP_GROUP);
        if (i == first_cap && cache && !have_ballot) GL_INS(&o, GL_CAP, GL_CAP_BALLOT);
        if (i == first_cap && pg_used && !have_i64) GL_INS(&o, GL_CAP, GL_CAP_INT64);   // LOCAL (AMD): OOT_GLOOP_PAGES
        if (i == first_cap && pg_used && !have_psb) GL_INS(&o, GL_CAP, GL_CAP_PSB);
        if (pg_used && w[1] < 0x00010500 && i == (first_imp != ~0u ? first_imp : first_mm)) {
            uint32_t ext[8] = {0};
            memcpy(ext, "SPV_KHR_physical_storage_buffer", 31);
            gl_ins(&o, GL_EXTENSION, 8, ext);
        }
        if (i == first_fn) {   // the new constants, then the array
            for (uint32_t q = 0; q < d->nnew; q++) GL_INS(&o, GL_CONSTANT, i32, d->newid[q], d->newv[q]);
            GL_INS(&o, GL_TYPE_ARRAY, arr_t, struct_t, cnv);
            GL_INS(&o, GL_TYPE_POINTER, arr_ptr_t, sclass, arr_t);
            GL_INS(&o, GL_VARIABLE, arr_ptr_t, arr, sclass);
            if (new_priv) GL_INS(&o, GL_TYPE_POINTER, new_priv, GL_SC_PRIVATE, i32);
            for (uint32_t x2 = 0; cache && x2 < nh; x2++) GL_INS(&o, GL_VARIABLE, ptr_priv, lastvar[x2], GL_SC_PRIVATE, cm1);
            if (sc_on) {   // LOCAL (AMD): OOT_GLOOP_SITECACHE: i32[32] of -1 per helper, each helper's function type with the site
                GL_INS(&o, GL_TYPE_ARRAY, sc_arrT, i32, sc_cslots);
                GL_INS(&o, GL_TYPE_POINTER, sc_arrPtrT, GL_SC_PRIVATE, sc_arrT);
                uint32_t cc[2 + SC_SLOTS];
                cc[0] = sc_arrT, cc[1] = sc_init;
                for (uint32_t q = 0; q < SC_SLOTS; q++) cc[2 + q] = cm1;
                gl_ins(&o, 44 /* OpConstantComposite */, 2 + SC_SLOTS, cc);
                for (uint32_t x2 = 0; x2 < nh; x2++) {
                    GL_INS(&o, GL_VARIABLE, sc_arrPtrT, sc_arr[x2], GL_SC_PRIVATE, sc_init);
                    uint32_t ft[SC_MAXP + 3], q = 0;
                    ft[q++] = sc_fnT[x2], ft[q++] = sc_ret[x2];
                    for (uint32_t p = 0; p < sc_np[x2]; p++) ft[q++] = sc_ptypes[x2][p];
                    ft[q++] = i32;
                    gl_ins(&o, 33 /* OpTypeFunction */, (int)q, ft);
                }
            }
            if (pg_used) {   // LOCAL (AMD): OOT_GLOOP_PAGES
                if (pg_new_u32) GL_INS(&o, GL_TYPE_INT, pg_new_u32, 32, 0);
                if (pg_new_u64) GL_INS(&o, GL_TYPE_INT, pg_new_u64, 64, 0);
                GL_INS(&o, GL_TYPE_POINTER, pg_ptr64, GL_SC_PSB, pg_u64);
                for (uint32_t q = 0; q < pg_nT; q++) GL_INS(&o, GL_TYPE_POINTER, pg_ptrT[q], GL_SC_PSB, pg_T[q]);
                GL_INS(&o, GL_CONSTANT, pg_u64, pg_c64z, 0, 0);
                GL_INS(&o, GL_CONSTANT, pg_u64, pg_cmask, (1u << GLOOP_PT_SHIFT) - 1, 0);
            }
            if (mc_used) {   // LOCAL (AMD): OOT_GLOOP_MISSCACHE: the windows, Private u64, empty at first
                GL_INS(&o, GL_TYPE_POINTER, mc_ptr, GL_SC_PRIVATE, pg_u64);
                GL_INS(&o, GL_CONSTANT, pg_u64, mc_max, 0xffffffffu, 0xffffffffu);
                for (uint32_t x2 = 0; x2 < nh; x2++) {
                    GL_INS(&o, GL_VARIABLE, mc_ptr, mc_lo[x2], GL_SC_PRIVATE, mc_max);
                    GL_INS(&o, GL_VARIABLE, mc_ptr, mc_hi[x2], GL_SC_PRIVATE, pg_c64z);
                }
            }
        }
        if (x < nh && i == hs[x].f0) {
            gl_put(&o, body[x].p, body[x].n);
            i = hs[x].f1, x++;
            continue;
        }
        if ((op == GL_VARIABLE && len >= 3 && a[1] < m.bound && kvar[a[1]]) ||
            ((op == GL_DECORATE || op == GL_NAME) && len >= 2 && a[0] < m.bound && kvar[a[0]])) {
        } else if (op == GL_ENTRY && len >= 4) {           // SPIR-V 1.4 and later list every global it uses
            const char *s = (const char *)(a + 2);
            uint32_t nl = (uint32_t)(strnlen(s, (size_t)(len - 3) * 4) / 4 + 1), q = 0, dropped = 0;
            if (len + 4 * nh > 250 || 2 + nl > len - 1) goto out;
            for (uint32_t y = 0; y < 2 + nl; y++) t[q++] = a[y];
            for (uint32_t y = 2 + nl; y < len - 1; y++) {
                if (a[y] < m.bound && kvar[a[y]]) dropped++;
                else t[q++] = a[y];
            }
            if (dropped) t[q++] = arr;
            for (uint32_t x2 = 0; cache && w[1] >= 0x00010400 && x2 < nh; x2++) t[q++] = lastvar[x2];   // 1.4: every global
            for (uint32_t x2 = 0; mc_used && w[1] >= 0x00010400 && x2 < nh; x2++) t[q++] = mc_lo[x2], t[q++] = mc_hi[x2];
            for (uint32_t x2 = 0; sc_on && w[1] >= 0x00010400 && x2 < nh; x2++) t[q++] = sc_arr[x2];
            gl_ins(&o, op, (int)q, t);
        } else if ((op == GL_ACCESS || op == GL_ACCESS_IB) && len >= 4 && a[2] < m.bound && kvar[a[2]]) {
            t[0] = a[0], t[1] = a[1], t[2] = arr, t[3] = d->cid[kvar[a[2]] - 1];
            for (uint32_t q = 3; q < len - 1; q++) t[q + 1] = a[q];
            gl_ins(&o, op, (int)len, t);
        } else if (op == GL_ARRAY_LENGTH && len == 5 && a[2] < m.bound && kvar[a[2]]) {
            GL_INS(&o, GL_ACCESS, ptr_t, use_ep[i], arr, d->cid[kvar[a[2]] - 1]);
            GL_INS(&o, GL_ARRAY_LENGTH, a[0], a[1], use_ep[i], a[3]);
        } else if (op == GL_MEMORY_MODEL && pg_used && len == 3) {   // LOCAL (AMD): OOT_GLOOP_PAGES: 64-bit addressing
            t[0] = GL_AM_PSB64, t[1] = a[1];
            gl_ins(&o, GL_MEMORY_MODEL, 2, t);
        } else {
            int hx = -1;   // LOCAL (AMD): OOT_GLOOP_SITECACHE: a call of a helper gets its site slot as a last argument
            if (sc_on && op == GL_CALL && len >= 4) for (uint32_t x2 = 0; x2 < nh; x2++) if (sc_fn[x2] == a[2]) hx = (int)x2;
            if (hx >= 0) {
                if (len > 250) goto out;
                for (uint32_t q = 0; q < len - 1; q++) t[q] = a[q];
                t[len - 1] = gl_c(d, sc_sites[hx]++ % SC_SLOTS);
                if (!t[len - 1]) goto out;
                gl_ins(&o, GL_CALL, (int)len, t);
            } else gl_put(&o, w + m.at[i], len);
        }
        if (i == last_dec) {
            GL_INS(&o, GL_DECORATE, arr, GL_DEC_SET, dset[vars[0]]);
            GL_INS(&o, GL_DECORATE, arr, GL_DEC_BINDING, binding[vars[0]]);
        }
    }
#undef GL_REN
    if (o.oom) goto out;
    o.p[3] = d->bound;
    *out_n = o.n, *first_binding = binding[vars[0]], *count = nv, *set = cset;
    res = o.p, o.p = NULL;
out:
    free(o.p);
    for (uint32_t x = 0; hs && x < GLOOP_MAX_HELPERS; x++) free(hs[x].c);
    for (uint32_t x = 0; body && x < GLOOP_MAX_HELPERS; x++) free(body[x].p);
    free(hs), free(body), free(d), free(use_ep);
    free(m.at), free(m.def), free(m.cval), free(m.cint), free(blk_of), free(map), free(ren), free(binding), free(dset), free(kvar);
    return res;
}
