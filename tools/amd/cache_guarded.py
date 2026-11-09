"""Guarded helpers of every module in a shader cache folder (or of the modules of given word counts), with their stage:
  python tools/amd/cache_guarded.py <shader-cache dir> [words ...]
Prints: words, stage, helpers (candidates each), calls, file. For the AMD investigation (tools/amd/spv_guarded.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spv_guarded as g  # noqa: E402

STAGES = {0: "VS", 4: "FS", 5: "CS", 5364: "TASK", 5365: "MESH", 3: "GS"}
folder, sizes = sys.argv[1], {int(x) for x in sys.argv[2:]}
for name in sorted(os.listdir(folder)):
    if not name.endswith(".spv"):
        continue
    path = os.path.join(folder, name)
    words = os.path.getsize(path) // 4
    if sizes and words not in sizes:
        continue
    header, insts = g.load(path)
    model = next((i[1] for i in insts if i[0] == 15), None)
    helpers = g.find_helpers(insts)
    # also count guarded-looking functions the shape test refuses: 2-4 pointer parameters and ArrayLength inside
    funcs = g.functions(insts)
    loose = 0
    for s, e in funcs:
        params = sum(1 for ins in insts[s:e] if ins[0] == g.OP_FUNCTION_PARAMETER)
        lengths = sum(1 for ins in insts[s:e] if ins[0] == g.OP_ARRAY_LENGTH)
        if 2 <= params <= 4 and lengths >= 2:
            loose += 1
    hid = {h[2] for h in helpers}
    calls = sum(1 for i in insts if i[0] == g.OP_FUNCTION_CALL and i[3] in hid)
    print(f"{words:8d} {STAGES.get(model, model)} helpers {len(helpers)} "
          f"({','.join(str(len(h[3][0])) for h in helpers)}) guarded-like functions {loose} calls {calls} {name[:16]}")
