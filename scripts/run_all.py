"""Build and validate the release, run isolated measurements, then play and audit the match."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from datasets import ROOT


def run(*arguments):
    subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-orig",action="store_true",help="Use the already built original and its object manifest")
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    if not args.keep_orig:
        run("build.py","orig")
    if (ROOT/"data/v2/release.json").is_file():
        directory=args.out or ROOT/"match_v2"
        run("scripts/v2_release.py","validate")
        run("scripts/benchmark.py","final","--reps",8)
        run("scripts/arena.py","--v2","--out",directory)
        run("scripts/v2_report.py","--match-dir",directory)
        sys.exit(0)
    args.out=args.out or ROOT/"match"
    run("build.py","plus","p1","p2","p3","p4","p123")
    run("scripts/source_diff.py")
    with ThreadPoolExecutor(4) as pool:
        tasks=[pool.submit(run,"scripts/"+s+".py") for s in
               ("probes","node_symbols","inspect_isa","codebook_profile")]
        for task in tasks:
            task.result()
    with ThreadPoolExecutor(3) as pool:
        tasks=[pool.submit(run,"scripts/"+s+".py") for s in ("bit_gate","tactics","mix10_gate")]
        for task in tasks:
            task.result()
    run("scripts/benchmark.py","final","--reps",8)
    run("scripts/arena.py","--out",args.out)
    run("scripts/report.py","--match-dir",args.out)
