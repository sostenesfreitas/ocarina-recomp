"""Merges bindings b0 .. b0+n-1 of set 0 of a dumped pipeline.txt into one binding of n descriptors, as the runtime's
OOT_GUARDED_LOOP layout does:  python tools/amd/merge_layout.py <in pipeline.txt> <out pipeline.txt> <b0> <n>"""
import re
import sys

src, dst, b0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out, cur, removed = [], None, 0
lines = open(src).read().splitlines()
for ln in lines:   # how many set 0 lines the merge removes
    if ln.startswith("set "):
        cur = int(ln.split()[1])
    m = re.match(r"binding (\d+) (\d+) (\d+) (\S+)$", ln)
    if m and cur == 0 and b0 < int(m.group(1)) < b0 + n:
        removed += 1
cur = None
for ln in lines:
    if ln.startswith("set "):
        cur = int(ln.split()[1])
    m = re.match(r"binding (\d+) (\d+) (\d+) (\S+)$", ln)
    if m and cur == 0 and b0 <= int(m.group(1)) < b0 + n:
        if int(m.group(1)) == b0:
            out.append(f"binding {b0} {m.group(2)} {n} {m.group(4)}")
        continue
    mm = re.match(r"set 0 (\S+) (\d+)$", ln)
    out.append(f"set 0 {mm.group(1)} {int(mm.group(2)) - removed}" if mm else ln)
open(dst, "w").write("\n".join(out) + "\n")
