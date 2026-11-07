"""The pipeline compile journal and quarantine (runtime/pipe_journal.c), across simulated game processes.

Each run of tests/pipe_journal.c is one game process: it opens the journal (reading what the previous run left), may
begin a compile and crash or die without ending it, and prints the strikes the quarantine gives its pipeline id. No
GPU, no game files.

    python tests/pipe_journal_test.py --cc <clang>
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
A, B, C, D, E = '00000000000000a1', '00000000000000b2', '00000000000000c3', '00000000000000d4', '00000000000000e5'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=shutil.which('clang') or 'clang')
    args = ap.parse_args()
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / ('pipe_journal.exe' if sys.platform == 'win32' else 'pipe_journal')
        subprocess.run([args.cc, '-O2', '-Wall', '-Werror', '-I', str(ROOT / 'runtime'), str(ROOT / 'runtime/pipe_journal.c'),
                        str(ROOT / 'tests/pipe_journal.c'), '-lpthread', '-o', str(exe)], check=True)
        work = tmp / 'build'
        work.mkdir()

        def run(mode, ident):
            r = subprocess.run([str(exe), mode, str(work), ident], capture_output=True, text=True, timeout=30)
            return r.stdout

        def check(name, ok, out=''):
            print(('ok   ' if ok else 'FAIL ') + name)
            if not ok:
                failures.append(name)
                if out:
                    print(out)

        def strikes(ident):
            out = run('strikes', ident)
            line = next((l for l in out.splitlines() if l.startswith('strikes ')), 'strikes ?')
            return line.split()[1], out

        out = run('strikes', A)
        check('a first run has no quarantine', 'strikes 0' in out, out)

        out = run('driver', A)
        check('the crash names the pipeline', f'graphics pipeline {A}' in out and 'on this thread' in out, out)
        s, out = strikes(A)
        check('a driver crash while compiling gives strike 1', s == '1', out)
        check('the next run logs the crash module', 'crashed in nvgpucomp64.dll' in out, out)

        out = run('driver', A)
        check('strike 1 compiles without optimization', '(already without optimization)' in out, out)
        s, out = strikes(A)
        check('a second crash gives strike 2', s == '2', out)

        s, out = strikes(A)
        check('strikes stay without new crashes', s == '2', out)

        run('killed', B)
        s, out = strikes(B)
        check('a run killed mid-compile gives no strike', s == '0' and 'no crash recorded' in out, out)

        run('clean', C)
        s, out = strikes(C)
        check('a crash after the compile ended blames nothing', s == '0', out)

        run('own', D)
        s, out = strikes(D)
        check('a crash in the game code does not blame a compile on another thread', s == '0', out)

        run('foreign', D)
        s, out = strikes(D)
        check('a crash in a DLL off the compiling thread blames every compile', s == '1', out)

        run('compute', E)
        s, out = strikes(E)
        q = (work / 'pipeline-quarantine.txt').read_text()
        check('compute pipelines get strikes too', s == '1' and f'{E} compute 1' in q, q)
        check('the quarantine file keeps the other ids', f'{A} graphics 2' in q and f'{D} graphics 1' in q, q)

        out = run('off', A)
        check('TOTK_PIPE_QUARANTINE=0 ignores the quarantine', 'strikes 0' in out and 'off' in out, out)

        lines = [l for l in q.splitlines() if not l.startswith(A)]
        (work / 'pipeline-quarantine.txt').write_text('\n'.join(lines) + '\n')
        s, out = strikes(A)
        check('deleting a line clears its strikes', s == '0', out)

        (work / 'pipeline-journal.bin').write_bytes(b'garbage')
        s, out = strikes(E)
        check('a damaged journal is ignored', s == '1', out)

    print(f'{len(failures)} failure(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
