"""Variants of a module's guarded storage loads, to find what makes AMD's compiler (LLPC) stall on them.

The translator (tools/xlate/patches/0015, 0025) reads guest memory it cannot tie to one storage buffer through a helper
per call site kind: int f(int *addr_lo, int *addr_hi) that tries each candidate buffer in turn,

    entry:  per candidate k a Function variable; then candidate 0's test
    test k: off = lo - base_k; hi ok; (off >> 2) < arrayLength(ssbo_k); off & 3 == 0
            OpSelectionMerge next_k; OpBranchConditional cond_k then_k next_k
    then k: return ssbo_k[off >> 2]
    next k: test k + 1 ... ; last: return 0

and main calls it once per load (hundreds of calls; LLPC inlines them all).

  python tools/amd/spv_guarded.py <in.spv> <out.spv> [--find] [--truncate N] [--calls N] [--dontinline] [--branchless]
    --find        list the helpers found and exit
    --truncate N  keep the first N candidates of each helper (the rest return 0): compile time against candidates
    --calls N     keep the first N calls of the helpers in the module (the others become 0): time against calls
    --dontinline  FunctionControl DontInline on the helpers
    --branchless  each helper becomes one block: every candidate's test and load, then a select chain (first match
                  wins, as before). Loads of the candidates that miss read out of bounds or a wrong word, which the
                  selects drop; robustBufferAccess keeps those reads harmless.
    --looparray [pipeline.txt out.txt]
                  the 32 candidates from the runtime's table at c0 + 0x800 (one storage buffer each, consecutive
                  bindings) become one loop over k: base from c0[128 + k], length and load through one storage buffer
                  array of 32 at the first of those bindings. The other candidates stay as they are, before the loop.
                  With pipeline.txt, writes the dump's description with those 32 bindings merged into one.
"""
import re
import struct
import sys

OP_FUNCTION, OP_FUNCTION_PARAMETER, OP_FUNCTION_END, OP_FUNCTION_CALL = 54, 55, 56, 57
OP_LABEL, OP_SELECTION_MERGE, OP_BRANCH_CONDITIONAL, OP_RETURN_VALUE = 248, 247, 250, 254
OP_VARIABLE, OP_SELECT, OP_COPY_OBJECT, OP_TYPE_BOOL, OP_CONSTANT, OP_TYPE_INT = 59, 169, 83, 20, 43, 21


def load(path):
    data = open(path, "rb").read()
    w = list(struct.unpack("<%dI" % (len(data) // 4), data))
    assert w[0] == 0x07230203, "not SPIR-V"
    out, i = [], 5
    while i < len(w):
        n = w[i] >> 16
        out.append([w[i] & 0xFFFF] + w[i + 1:i + n])
        i += n
    return w[:5], out


def save(path, header, insts):
    words = list(header)
    for ins in insts:
        words.append(len(ins) << 16 | ins[0])
        words.extend(ins[1:])
    open(path, "wb").write(struct.pack("<%dI" % len(words), *words))


def functions(insts):
    """(start, end) index ranges of OpFunction .. OpFunctionEnd."""
    out, start = [], None
    for k, ins in enumerate(insts):
        if ins[0] == OP_FUNCTION:
            start = k
        elif ins[0] == OP_FUNCTION_END:
            out.append((start, k))
    return out


def blocks(insts, s, e):
    """Blocks of the function at [s, e]: list of (label id, [instructions from OpLabel to the terminator])."""
    out, cur = [], None
    for k in range(s, e):
        ins = insts[k]
        if ins[0] == OP_LABEL:
            cur = (ins[1], [ins])
            out.append(cur)
        elif cur is not None:
            cur[1].append(ins)
    return out


def helper_shape(insts, s, e):
    """Candidates [(test block instructions, cond id, then block instructions, value id)] and the final return value id
    when the function has the guarded-load shape; else None."""
    params = [ins for ins in insts[s:e] if ins[0] == OP_FUNCTION_PARAMETER]
    if len(params) != 2:
        return None
    bl = blocks(insts, s, e)
    if len(bl) < 3:
        return None
    by_id = {lab: body for lab, body in bl}
    cands, cur = [], bl[0][1]
    seen = set()
    while True:
        term = cur[-1]
        if term[0] == OP_RETURN_VALUE:
            return cands, term[1]
        if term[0] != OP_BRANCH_CONDITIONAL or len(cur) < 2 or cur[-2][0] != OP_SELECTION_MERGE:
            return None
        cond, then_id, next_id = term[1], term[2], term[3]
        if cur[-2][1] != next_id or then_id not in by_id or next_id not in by_id or next_id in seen:
            return None
        then = by_id[then_id]
        if then[-1][0] != OP_RETURN_VALUE:
            return None
        cands.append((cur, cond, then, then[-1][1]))
        seen.add(next_id)
        cur = by_id[next_id]


def find_helpers(insts):
    found = []
    for s, e in functions(insts):
        shape = helper_shape(insts, s, e)
        if shape and len(shape[0]) >= 4:
            found.append((s, e, insts[s][2], shape))
    return found


OP_NAME, OP_DECORATE, OP_ACCESS_CHAIN, OP_ARRAY_LENGTH, OP_TYPE_ARRAY, OP_TYPE_POINTER = 5, 71, 65, 68, 28, 32
OP_PHI, OP_LOOP_MERGE, OP_BRANCH, OP_IADD, OP_SLESS_THAN, OP_STORE, OP_LOAD = 245, 246, 249, 128, 177, 62, 61
DEC_BINDING, DEC_SET = 33, 34
DYN_N, DYN_VEC0 = 32, 0x800 // 16


def lit_str(ops):
    b = b"".join(struct.pack("<I", x) for x in ops)
    return b.split(b"\0", 1)[0].decode("utf-8", "replace")


class Rewriter:
    """Loop-and-array rewrite of the c0 + 0x800 candidates (--looparray). Raises ValueError when the module is not
    exactly the shape the translator emits, so nothing is changed then."""

    def __init__(self, header, insts):
        self.header, self.insts = header, insts
        self.bound = header[3]
        self.names = {i[1]: lit_str(i[2:]) for i in insts if i[0] == OP_NAME}
        self.const = {}       # (type, value) -> id
        self.defs = {}
        for i in insts:
            if i[0] == OP_CONSTANT and len(i) == 4:
                self.const[(i[1], i[3])] = i[2]
        self.i32 = next(i[1] for i in insts if i[0] == OP_TYPE_INT and i[2] == 32 and i[3] == 1)
        self.boolt = next(i[1] for i in insts if i[0] == OP_TYPE_BOOL)
        self.new_globals = []  # inserted before the first OpFunction
        self.new_decor = []

    def newid(self):
        self.bound += 1
        return self.bound - 1

    def c(self, v):
        k = (self.i32, v & 0xFFFFFFFF)
        if k not in self.const:
            nid = self.newid()
            self.const[k] = nid
            self.new_globals.append([OP_CONSTANT, self.i32, nid, v & 0xFFFFFFFF])
        return self.const[k]


def pointer_operands(ins):
    """Operand positions (in ins) that may name a pointer, for the instructions that take one."""
    op, n = ins[0], len(ins)
    if op in (OP_LOAD, 65, 66, 67, 70, OP_ARRAY_LENGTH, 83, 60):        # Load, access chains, ArrayLength, CopyObject, ImageTexelPointer
        return [3] if n > 3 else []
    if op in (OP_STORE, 63, 64):                                       # Store, CopyMemory(Sized)
        return [1, 2] if op != OP_STORE else [1]
    if op == OP_FUNCTION_CALL:
        return list(range(4, n))
    if 227 <= op <= 242:                                               # atomics: pointer first
        return [3] if op != 228 else [1]
    if op in (OP_PHI, OP_SELECT, 401, 402):
        return list(range(3, n))
    if op == 15:                                                       # entry point interface
        return list(range(3, n))
    return []


def candidate_info(rw, test, then, c0var):
    """(vec index constant id, ssbo variable, function variable) of a c0-table candidate, or None."""
    vec = None
    for ins in test:
        if ins[0] == OP_ACCESS_CHAIN and len(ins) == 7 and ins[3] == c0var:
            vec = ins[5] if vec is None or vec == ins[5] else -1
    if vec is None or vec == -1:
        return None
    ssbo = [ins[3] for ins in then if ins[0] == OP_ACCESS_CHAIN]
    stores = [ins[1] for ins in test if ins[0] == OP_STORE]
    if len(ssbo) != 1 or len(stores) != 1:
        return None
    return vec, ssbo[0], stores[0]


def canonical(code, subst):
    """Instructions with result ids numbered locally and the per-candidate ids replaced by names."""
    local, out = {}, []
    for ins in code:
        op = ins[0]
        has_res = op not in (OP_STORE, OP_SELECTION_MERGE, OP_BRANCH_CONDITIONAL, OP_RETURN_VALUE, OP_LABEL)
        row = [op]
        for n, x in enumerate(ins[1:], 1):
            if has_res and n == 2:
                local[x] = f"#{len(local)}"
                row.append(local[x])
            elif (op == OP_SELECTION_MERGE and n >= 1) or (op == OP_BRANCH_CONDITIONAL and n >= 2):
                row.append("L")
            else:
                row.append(subst.get(x, local.get(x, x)))
        out.append(tuple(row))
    return out


def looparray(rw, helpers, pipeline_in=None, pipeline_out=None):
    insts = rw.insts
    c0vars = [v for v, n in rw.names.items() if re.fullmatch(r"[a-z]+_c0", n)]
    if len(c0vars) != 1:
        raise ValueError("no single <stage>_c0 uniform buffer")
    c0var = c0vars[0]
    ssbo_of_k = {}
    plan = []
    for s, e, fid, (cands, last) in helpers:
        info = [candidate_info(rw, t[1:] if t[0][0] == OP_LABEL else t, th, c0var) for t, cond, th, val in cands]
        # the trailing run of c0-table candidates, k = 0..31 in order
        first = len(cands) - DYN_N
        if first < 0:
            raise ValueError(f"%{fid}: fewer than {DYN_N} candidates")
        consts = {v: k for (t, k), v in rw.const.items() if t == rw.i32}
        for j in range(first, len(cands)):
            if info[j] is None or consts.get(info[j][0]) != DYN_VEC0 + (j - first):
                raise ValueError(f"%{fid}: candidate {j} is not c0 table entry {j - first}")
            k = j - first
            if ssbo_of_k.setdefault(k, info[j][1]) != info[j][1]:
                raise ValueError(f"%{fid}: entry {k} uses another storage buffer than in another helper")
        # the same code for every entry, up to its constant, buffer and variable
        ref = None
        for j in range(first, len(cands)):
            t, cond, th, val = cands[j]
            body = [x for x in (t[1:] if t[0][0] == OP_LABEL else t) if x[0] != OP_VARIABLE]
            if j == 0:
                body = [x for x in body]
            vec, sb, fv = info[j]
            canon = canonical(body + th[1:], {vec: "VEC", sb: "SSBO", fv: "FVAR"})
            if ref is None:
                ref = canon
            elif canon != ref:
                raise ValueError(f"%{fid}: candidate {j} differs from the first table candidate")
        plan.append((s, e, fid, cands, last, first, info))
    # the storage buffers: one variable per entry, consecutive bindings, one struct type
    var_type, binding, dset = {}, {}, {}
    for ins in insts:
        if ins[0] == OP_VARIABLE:
            var_type[ins[2]] = ins[1]
        if ins[0] == OP_DECORATE and ins[2] == DEC_BINDING:
            binding[ins[1]] = ins[3]
        if ins[0] == OP_DECORATE and ins[2] == DEC_SET:
            dset[ins[1]] = ins[3]
    sb = [ssbo_of_k[k] for k in range(DYN_N)]
    if len({var_type[v] for v in sb}) != 1 or len({dset.get(v, 0) for v in sb}) != 1:
        raise ValueError("the table's storage buffers differ in type or set")
    b0 = binding[sb[0]]
    if [binding[v] for v in sb] != list(range(b0, b0 + DYN_N)):
        raise ValueError("the table's storage buffers are not on consecutive bindings")
    ptr_t = var_type[sb[0]]
    ptr_ins = next(i for i in insts if i[0] == OP_TYPE_POINTER and i[1] == ptr_t)
    sclass, struct_t = ptr_ins[2], ptr_ins[3]
    # new globals: array type, its pointer, the variable
    c32 = rw.c(DYN_N)
    arr_t, arr_ptr_t, arr = rw.newid(), rw.newid(), rw.newid()
    rw.new_globals += [[OP_TYPE_ARRAY, arr_t, struct_t, c32], [OP_TYPE_POINTER, arr_ptr_t, sclass, arr_t],
                       [OP_VARIABLE, arr_ptr_t, arr, sclass]]
    rw.new_decor += [[OP_DECORATE, arr, DEC_SET, dset.get(sb[0], 0)], [OP_DECORATE, arr, DEC_BINDING, b0]]
    k_of_var = {v: k for k, v in enumerate(sb)}
    cz, c1, cvec0 = rw.c(0), rw.c(1), rw.c(DYN_VEC0)
    # rebuild each helper (last first, so earlier indices stay valid)
    for s, e, fid, cands, last, first, info in sorted(plan, key=lambda p: -p[0]):
        head = [ins for ins in insts[s:e] if ins[0] in (OP_FUNCTION, OP_FUNCTION_PARAMETER)]
        entry = cands[0][0]
        variables = [ins for ins in entry if ins[0] == OP_VARIABLE]
        body = [entry[0]] + variables
        for j in range(first):   # the other candidates, unchanged
            t, cond, th, val = cands[j]
            body += [x for x in t if x[0] != OP_VARIABLE and x is not entry[0]]
            body += th
        if first:
            pred = cands[first][0][0][1]          # the label the last of them branches to
            body += [[OP_LABEL, pred]]
        else:
            pred = entry[0][1]
        H, Body, Hit, SM, Cont, Exit = (rw.newid() for _ in range(6))
        k, k1, more, vidx = rw.newid(), rw.newid(), rw.newid(), rw.newid()
        body += [[OP_BRANCH, H],
                 [OP_LABEL, H], [OP_PHI, rw.i32, k, cz, pred, k1, Cont], [OP_LOOP_MERGE, Exit, Cont, 2], [OP_BRANCH, Body],
                 [OP_LABEL, Body], [OP_IADD, rw.i32, vidx, k, cvec0]]
        t, cond, th, val = cands[first]
        vec, sb0, fv = info[first]
        ren = {}

        def clone(ins):
            op = ins[0]
            if op == OP_ACCESS_CHAIN and ins[3] == c0var:
                r = rw.newid(); ren[ins[2]] = r
                return [[op, ins[1], r, c0var] + [vidx if x == vec else ren.get(x, x) for x in ins[4:]]]
            if op == OP_ACCESS_CHAIN and ins[3] == sb0:
                r = rw.newid(); ren[ins[2]] = r
                return [[op, ins[1], r, arr, k] + [ren.get(x, x) for x in ins[4:]]]
            if op == OP_ARRAY_LENGTH and ins[3] == sb0:
                ep, r = rw.newid(), rw.newid(); ren[ins[2]] = r
                return [[OP_ACCESS_CHAIN, ptr_t, ep, arr, k], [OP_ARRAY_LENGTH, ins[1], r, ep, ins[4]]]
            if op in (OP_STORE,):
                return [[op] + [ren.get(x, x) for x in ins[1:]]]
            r = rw.newid(); ren[ins[2]] = r
            return [[op, ins[1], r] + [ren.get(x, x) for x in ins[3:]]]

        test = [x for x in (t[1:] if t[0][0] == OP_LABEL else t) if x[0] != OP_VARIABLE][:-2]
        for ins in test:
            body += clone(ins)
        body += [[OP_SELECTION_MERGE, SM, 0], [OP_BRANCH_CONDITIONAL, ren[cond], Hit, SM], [OP_LABEL, Hit]]
        for ins in th[1:-1]:
            body += clone(ins)
        body += [[OP_RETURN_VALUE, ren[val]],
                 [OP_LABEL, SM], [OP_BRANCH, Cont],
                 [OP_LABEL, Cont], [OP_IADD, rw.i32, k1, k, c1], [OP_SLESS_THAN, rw.boolt, more, k1, c32],
                 [OP_BRANCH_CONDITIONAL, more, H, Exit],
                 [OP_LABEL, Exit], [OP_RETURN_VALUE, last]]
        insts[s:e + 1] = head + body + [[OP_FUNCTION_END]]
    # other uses of the 32 variables: through the array with a constant index
    out = []
    for ins in insts:
        if ins[0] in (OP_ACCESS_CHAIN, 66) and len(ins) > 3 and ins[3] in k_of_var:
            out.append([ins[0], ins[1], ins[2], arr, rw.c(k_of_var[ins[3]])] + ins[4:])
        elif ins[0] == OP_ARRAY_LENGTH and ins[3] in k_of_var:
            ep = rw.newid()
            out += [[OP_ACCESS_CHAIN, ptr_t, ep, arr, rw.c(k_of_var[ins[3]])], [OP_ARRAY_LENGTH, ins[1], ins[2], ep, ins[4]]]
        elif ins[0] == OP_VARIABLE and ins[2] in k_of_var:
            continue
        elif ins[0] in (OP_DECORATE, OP_NAME) and ins[1] in k_of_var:
            continue
        else:
            if any(ins[n] in k_of_var for n in pointer_operands(ins)):
                raise ValueError(f"op {ins[0]} uses a table storage buffer another way")
            out.append(ins)
    # dynamically uniform indices into an array of storage buffers
    if not any(i[0] == 17 and i[1] == 30 for i in out):
        out.insert(next(n for n, i in enumerate(out) if i[0] == 17), [17, 30])
    # globals before the first function, decorations after the last decoration
    fpos = next(n for n, i in enumerate(out) if i[0] == OP_FUNCTION)
    out[fpos:fpos] = rw.new_globals
    dpos = max(n for n, i in enumerate(out) if i[0] in (OP_DECORATE, 72)) + 1
    out[dpos:dpos] = rw.new_decor
    rw.insts[:] = out
    rw.header[3] = rw.bound
    if pipeline_in:
        stage_bits = None
        lines = open(pipeline_in).read().splitlines()
        res, cur_set = [], None
        for ln in lines:
            m = re.match(r"binding (\d+) (\d+) (\d+) (0x[0-9a-f]+|\d+)", ln)
            if ln.startswith("set "):
                cur_set = int(ln.split()[1])
            if m and cur_set == dset.get(sb[0], 0) and b0 <= int(m.group(1)) < b0 + DYN_N:
                if int(m.group(1)) == b0:
                    res.append(f"binding {b0} {m.group(2)} {DYN_N} {m.group(4)}")
                continue
            res.append(ln)
        res = [re.sub(r"^set (\d+) (\S+) (\d+)$", lambda mm: f"set {mm.group(1)} {mm.group(2)} {int(mm.group(3)) - (DYN_N - 1)}"
                      if int(mm.group(1)) == dset.get(sb[0], 0) else mm.group(0), x) for x in res]
        open(pipeline_out, "w").write("\n".join(res) + "\n")
    return b0


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    src, dst, opts = a[0], a[1], a[2:]
    header, insts = load(src)
    helpers = find_helpers(insts)
    hid = {h[2] for h in helpers}
    ncalls = sum(1 for ins in insts if ins[0] == OP_FUNCTION_CALL and ins[3] in hid)
    print(f"{src}: {len(helpers)} guarded helpers, {ncalls} calls: " +
          ", ".join(f"%{h[2]} ({len(h[3][0])} candidates)" for h in helpers))
    if "--find" in opts:
        return
    bound = header[3]
    i32 = next((ins[1] for ins in insts if ins[0] == OP_TYPE_INT and ins[2] == 32 and ins[3] == 1), None)
    zero = next((ins[2] for ins in insts if ins[0] == OP_CONSTANT and ins[1] == i32 and ins[3] == 0), None)

    def newid():
        nonlocal bound
        bound += 1
        return bound - 1

    # --calls N: later calls return 0
    if "--calls" in opts:
        keep = int(opts[opts.index("--calls") + 1])
        n = 0
        for k, ins in enumerate(insts):
            if ins[0] == OP_FUNCTION_CALL and ins[3] in hid:
                n += 1
                if n > keep:
                    assert zero is not None, "no int 0 constant"
                    insts[k] = [OP_COPY_OBJECT, ins[1], ins[2], zero]
    # --looparray [pipeline.txt out.txt]
    if "--looparray" in opts:
        n = opts.index("--looparray")
        pin = opts[n + 1] if len(opts) > n + 2 and not opts[n + 1].startswith("--") else None
        pout = opts[n + 2] if pin else None
        rw = Rewriter(header, insts)
        try:
            b0 = looparray(rw, helpers, pin, pout)
        except ValueError as e:
            sys.exit(f"--looparray: not rewritten: {e}")
        save(dst, rw.header, rw.insts)
        print(f"wrote {dst}: the c0 table candidates loop over one storage buffer array at binding {b0}" +
              (f"; {pout}" if pout else ""))
        return
    # --dontinline
    if "--dontinline" in opts:
        for s, e, fid, shape in helpers:
            insts[s][3] = (insts[s][3] & ~1) | 2
    # rewrites of the helper bodies, last helper first so the indices stay valid
    trunc = int(opts[opts.index("--truncate") + 1]) if "--truncate" in opts else None
    if trunc is not None or "--branchless" in opts:
        boolt = next(ins[1] for ins in insts if ins[0] == OP_TYPE_BOOL)
        for s, e, fid, (cands, last) in sorted(helpers, key=lambda h: -h[0]):
            rtype = insts[s][1]
            head = [ins for ins in insts[s:e] if ins[0] in (OP_FUNCTION, OP_FUNCTION_PARAMETER)]
            if trunc is not None:
                cands = cands[:trunc]
            entry_label = cands[0][0][0]
            variables = [ins for ins in cands[0][0] if ins[0] == OP_VARIABLE]
            body = [entry_label] + variables
            if "--branchless" in opts:
                for test, cond, then, val in cands:
                    body += [ins for ins in test[1:-2] if ins[0] != OP_VARIABLE and ins[0] != OP_LABEL]
                    body += then[1:-1]
                acc = zero if trunc is not None else last
                for test, cond, then, val in reversed(cands):
                    r = newid()
                    body.append([OP_SELECT, rtype, r, cond, val, acc])
                    acc = r
                body.append([OP_RETURN_VALUE, acc])
            else:   # truncate, keeping the branches
                for j, (test, cond, then, val) in enumerate(cands):
                    t = [ins for ins in test if ins[0] != OP_VARIABLE] if j == 0 else list(test)
                    if j == 0:
                        t = t[1:]   # entry label already in body
                    body += t
                    body += then
                # the last test's "next" block: return 0
                nxt = cands[-1][0][-1][3]
                body += [[OP_LABEL, nxt], [OP_RETURN_VALUE, zero]]
            insts[s:e + 1] = head + body + [[OP_FUNCTION_END]]
    header[3] = bound
    save(dst, header, insts)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
