"""Save selected disassembly from the actual AVX2 policy and move functions."""
import re
import subprocess
from bit_gate import CDB
from datasets import ROOT, dump


def inspect():
    summary = {}
    for variant in ("orig","plus"):
        exe = ROOT/f"pbrain-rapfi-{variant}.exe"
        module = f"pbrain_rapfi_{variant}"
        prefix = f"{module}!Evaluation::mix9svq::Accumulator::"
        cmd = f".echo POLICY_START; uf {prefix}evaluatePolicy; .echo MOVE_START; uf {prefix}move; .echo ISA_END; q"
        asm = subprocess.check_output([CDB, "-sins", "-y", str(ROOT), "-c", cmd, str(exe)],
                                      cwd=ROOT,encoding="utf-8",errors="replace",timeout=35)
        policy,move = re.search(r"(?ms)^POLICY_START\s*$\n(.*?)^MOVE_START\s*$\n(.*?)^ISA_END\s*$",asm).groups()
        counts = dict(vpmaddwd=len(re.findall(r"\bvpmaddwd\b",policy)),
                      fused_float=len(re.findall(r"\bvfmadd\d+ps\b",policy)),
                      prefetcht0=len(re.findall(r"\bprefetcht0\b",move)))
        assert min(counts.values()) > 0,(variant,counts)
        (ROOT/f"validation/{variant}_policy_move.asm").write_text(policy+"\n"+move,encoding="utf-8")
        summary[variant]=counts
    dump(ROOT/"validation/isa.json",summary)
    print("Playing ISA",summary,flush=True)


if __name__ == "__main__":
    inspect()
