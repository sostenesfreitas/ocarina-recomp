"""Compile and exercise the production camera hooks without game data or a window."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    compiler = shutil.which('clang')
    if not compiler and os.name == 'nt':
        candidate = Path('C:/msys64/ucrt64/bin/clang.exe')
        if candidate.is_file():
            compiler = str(candidate)
    if not compiler:
        raise SystemExit('clang is required')
    out = ROOT / 'build/tests/camera-mouse'
    out.mkdir(parents=True, exist_ok=True)
    exe = out / ('test.exe' if os.name == 'nt' else 'test')
    env = os.environ.copy()
    env['TOTK_INPUT'] = 'none'
    env['PATH'] = str(Path(compiler).parent) + os.pathsep + env.get('PATH', '')
    subprocess.run([compiler, '-O2', '-fno-strict-aliasing', '-Wall', '-Wextra', '-I', str(ROOT / 'runtime'),
                    str(ROOT / 'tests/camera_mouse_test.c'), str(ROOT / 'runtime/input_camera.c'),
                    '-o', str(exe)] + ([] if os.name == 'nt' else ['-lm']), check=True, env=env)
    subprocess.run([str(exe)], check=True, env=env)
    if '--generated' in sys.argv[1:]:
        source = ROOT / 'generated/oot111/main/main_0078.c'
        if not source.is_file():
            raise SystemExit('The generated helpers from your own local game copy are required for --generated')
        text = source.read_text()
        bodies = []
        for address in ('7000623ac0', '7000623e70'):
            start = text.index(f'void f_{address}(Cpu *c) {{')
            end = text.index('\n}\n', start) + 3
            body = text[start:end]
            if 'ov_camera_axis_mouse(oot_mouse_node' not in body:
                raise SystemExit('Run tools/oot/patch_generated.py on the local generated sources first')
            bodies.append(body)
        derived = out / 'generated-axis-helpers.c'
        derived.write_text(text.splitlines()[0] + '\n#include "cpu.h"\n#include "simd.h"\n'
                           'void f_7000623b60(Cpu *c);\nvoid f_7000623f10(Cpu *c);\n'
                           'float ov_camera_axis_mouse(uint64_t, int, float, float);\n' + '\n'.join(bodies))
        exe = out / ('generated-test.exe' if os.name == 'nt' else 'generated-test')
        subprocess.run([compiler, '-O1', '-fno-strict-aliasing', '-march=x86-64-v3', '-I', str(ROOT / 'runtime'),
                        str(ROOT / 'tests/camera_mouse_generated_test.c'), str(derived),
                        str(ROOT / 'runtime/input_camera.c'), '-o', str(exe)] +
                       ([] if os.name == 'nt' else ['-lm']), check=True, env=env)
        subprocess.run([str(exe)], check=True, env=env)


if __name__ == '__main__':
    main()
