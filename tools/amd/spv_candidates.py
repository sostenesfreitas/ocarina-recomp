"""Lists the candidates of each guarded helper of a module: where its base is read (constant buffer, vec4, components)
and the binding of its storage buffer. For the AMD investigation (tools/amd/spv_guarded.py finds the helpers).

  python tools/amd/spv_candidates.py <file.spv>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spv_guarded as g  # noqa: E402


def main():
    header, insts = g.load(sys.argv[1])
    names = {i[1]: g.lit_str(i[2:]) for i in insts if i[0] == g.OP_NAME}
    consts = {i[2]: i[3] for i in insts if i[0] == g.OP_CONSTANT and len(i) == 4}
    binding = {i[1]: i[3] for i in insts if i[0] == g.OP_DECORATE and i[2] == g.DEC_BINDING}
    for s, e, fid, (cands, last) in g.find_helpers(insts):
        print(f"helper %{fid}: {len(cands)} candidates")
        runs = []
        for j, (t, cond, th, val) in enumerate(cands):
            reads = [ins for ins in t if ins[0] == g.OP_ACCESS_CHAIN and len(ins) == 7]
            ssbo = next((ins[3] for ins in t if ins[0] == g.OP_ARRAY_LENGTH), None)
            r = [(names.get(x[3], x[3]), consts.get(x[5]), consts.get(x[6])) for x in reads]
            key = (r[0][0], r[0][2], r[1][2]) if len(r) >= 2 else None
            vec = r[0][1] if r else None
            if runs and runs[-1][0] == key and vec == runs[-1][2] + runs[-1][3]:
                runs[-1][3] += 1
            else:
                runs.append([key, j, vec, 1, binding.get(ssbo)])
        for key, j, vec, n, b in runs:
            print(f"  from candidate {j}: {n} x base {key[0] if key else '?'}[vec {vec}..{vec + n - 1}] "
                  f"components {key[1:] if key else '?'}, first buffer binding {b}")


if __name__ == "__main__":
    main()
