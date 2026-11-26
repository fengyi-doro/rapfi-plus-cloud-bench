"""Read raw heads from the unmodified playing PEs with CDB and their own PDBs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
from datasets import DATA, ROOT, dump, sha
from rules import bits, forbidden

OUT = ROOT / "validation"
CDB = shutil.which("cdb") or str(Path(os.environ.get("ProgramFiles(x86)", ""))
                                / "Windows Kits/10/Debuggers/x64/cdb.exe")
ADDR = r"[0-9a-f]{8}`[0-9a-f]{8}"


def board_command(moves):
    return "\n".join(["YXBOARD", *(f"{p % 15},{p // 15},{1 if i % 2 == len(moves) % 2 else 2}"
                                   for i, p in enumerate(moves)), "DONE"])


def debugger_commands(exe, module, architecture):
    policy_member = "evaluatePolicyLarge" if architecture == "mix10" else "evaluatePolicy"
    names = [f"{module}!Evaluation::{architecture}::" + s for s in
             ("Evaluator::evaluateValue", "Accumulator::"+policy_member)]
    lookup = "; ".join(f".echo FUNCTION{i}; uf {n}; .echo END{i}" for i, n in enumerate(names))
    cmd = f"dt {module}!Evaluation::PolicyBuffer; dt {module}!Evaluation::ValueType; {lookup}; q"
    disasm = subprocess.check_output([CDB, "-sins", "-y", str(exe.parent), "-c", cmd, str(exe)],
                                     cwd=ROOT, encoding="utf-8", errors="replace", timeout=30)
    offset = int(re.search(r"\+0x([0-9a-f]+) policy\s+: \[\d+\] Float", disasm)[1], 16)
    assert all(re.search(rf"\+0x{off:03x} {name}\s+:", disasm)
               for off, name in [(0, "val"), (4, "winProb"), (8, "lossProb"), (12, "drawProb")])
    cmds = [f'bp {module}!Command::GomocupProtocol::traceSearch ".echo HEAD_POSITION; gc"',
            f'bp {names[1]} "r @$t0=@r8; gc"']
    for i, name in enumerate(names):
        section = re.search(rf"(?ms)^FUNCTION{i}\s*$\n(.*?)^END{i}\s*$", disasm)[1]
        start = int(re.search(rf"(?m)^({ADDR}) ", section)[1].replace("`", ""), 16)
        returns = re.findall(rf"(?m)^({ADDR})\s+c3\s+ret\s*$", section)
        assert returns, name
        for address in returns:
            delta = int(address.replace("`", ""), 16) - start
            command = (".echo HEAD_WDL; dd @rax L4" if i == 0 else
                       f".echo HEAD_POLICY; dd @$t0+0x{offset:x} L0xe1")
            cmds.append(f'bp {name}+0x{delta:x} "{command}; .echo HEAD_END; gc"')
    cmds += [".echo OBSERVER_READY", "g"]
    return "; ".join(cmds)


def capture(variant, rows, tag, architecture="mix9svq", config=ROOT/"config.toml", exe=None, out=None):
    exe = exe or ROOT / f"pbrain-rapfi-{variant}.exe"
    out = out or OUT
    module = exe.stem.replace("-", "_")
    commands = debugger_commands(exe, module, architecture)
    log = out / f"{tag}_{variant}_debug.log"
    cmd = [CDB, "-sins", "-G", "-y", str(exe.parent), "-c", commands]
    engine = subprocess.Popen([str(exe), "--config", str(config)],
                              cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    dbg = None
    try:
        with log.open("wb") as trace:
            dbg = subprocess.Popen(cmd + ["-p", str(engine.pid)], stdout=trace, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if re.search(rb"(?m)^OBSERVER_READY\r?$", log.read_bytes()):
                    break
                if dbg.poll() is not None:
                    raise RuntimeError(log.read_text(errors="replace")[-2000:])
                time.sleep(.03)
            else:
                raise TimeoutError(log.read_text(errors="replace")[-1800:])
            cmds = ["START 15", "INFO thread_num 1", "INFO pondering 0", "INFO hash_size 8192"]
            rule = None
            for row in rows:
                if row["rule"] != rule:
                    rule = row["rule"]
                    cmds.append(f"INFO rule {rule}")
                cmds += [board_command(row["moves"]), "TRACESEARCH", "TRACEBOARD"]
            cmds.append("END")
            protocol = engine.communicate(("\n".join(cmds) + "\n").encode(), timeout=150)[0]
            if engine.returncode or b"ERROR " in protocol:
                raise RuntimeError(protocol.decode(errors="replace")[-1600:])
            dbg.wait(timeout=15)
    finally:
        for proc in (dbg, engine):
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
    (out / f"{tag}_{variant}_protocol.log").write_bytes(protocol)
    text = log.read_text(errors="replace")
    sections = re.split(r"(?m)^HEAD_POSITION\s*$", text)[1:]
    static = re.findall(rb"MESSAGE Static Eval\[Black\]: (-?\d+)", protocol)
    evals = re.findall(rb"MESSAGE Evaluator Eval\[Black\]: (-?\d+)", protocol)
    assert len(static) == len(evals) == len(sections) == len(rows), (len(static), len(sections), len(rows))
    results = []
    for row, section, score, nn in zip(rows, sections, static, evals, strict=True):
        def packets(key, size):
            payloads = re.findall(rf"(?ms)^HEAD_{key}\s*$\n(.*?)^HEAD_END\s*$", section)
            values = [[int(word, 16) for line in packet.splitlines()
                       if re.match(ADDR + " ", line) for word in line.split()[1:]]
                      for packet in payloads]
            assert all(len(v) == size for v in values), (key, row.get("id"), [len(v) for v in values])
            return values
        wdl = packets("WDL", 4)
        policy = packets("POLICY", 225)
        assert wdl and len(policy) == 2 and all(v == wdl[-1] for v in wdl), (len(wdl), len(policy))
        v = struct.unpack("<i", struct.pack("<I", wdl[-1][0]))[0]
        side = len(row["moves"]) % 2
        assert int(nn) == v * (-1 if side else 1)
        occupied = set(row["moves"])
        stones = bits(row["moves"])
        empty = [p for p in range(225) if p not in occupied]
        sides = []
        legal_counts = []
        for index, color in enumerate((side, 1-side)):
            legal = [p for p in empty if not forbidden(stones[color], stones[1-color], p, row["rule"], color)]
            sides.append([policy[index][p] for p in empty])
            legal_counts.append(len(legal))
        results.append(dict(static_black=int(score), nn_stm=v, wdl=wdl[-1][1:], policy=sides,
                            legal=legal_counts, empty=len(empty)))
    return results


def run(candidate="plus", out=None):
    global OUT
    OUT = out or OUT
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ("orig", candidate)
    rows = json.loads((DATA / "positions.json").read_bytes())
    results = {v: capture(v, rows, "heads") for v in variants}
    for v, result in results.items():
        dump(OUT / f"heads_{v}.json", result)
    for row, a, b in zip(rows, results["orig"], results[candidate], strict=True):
        assert a == b, (row["id"], row["rule"], "raw head mismatch")
    result = dict(positions=len(rows), rules={str(r):sum(row["rule"] == r for row in rows) for r in range(3)},
                  wdl_words=3 * len(rows), both_color_empty_scores=2 * sum(r["empty"] for r in results["orig"]),
                  stm_legal_scores=sum(r["legal"][0] for r in results["orig"]),
                  both_color_legal_scores=sum(sum(r["legal"]) for r in results["orig"]),
                  policy_tail_residues=sorted({r["empty"] % 4 for r in results["orig"]}),
                  mismatches=0, exe={v:sha(ROOT/f"pbrain-rapfi-{v}.exe") for v in results},
                  raw_sha256={v:sha(OUT/f"heads_{v}.json") for v in results})
    assert result["both_color_empty_scores"] and result["policy_tail_residues"] == [0,1,2,3]
    dump(OUT / "heads.json", result)
    classical(len(rows), candidate)
    print("BIT GATE", result, flush=True)


def classical(count, candidate="plus"):
    packets = {}
    for variant in ("orig",candidate):
        lines = (OUT/f"heads_{variant}_protocol.log").read_text(encoding="utf-8").splitlines()
        static = [int(m[1]) for line in lines if (m:=re.fullmatch(r"MESSAGE Eval\[Black\]: (-?\d+)",line))]
        grids = [lines[i+1:i+16] for i,line in enumerate(lines)
                 if re.fullmatch(r"MESSAGE -+Score-+(Black|White)-+",line)]
        assert len(static)==count and len(grids)==2*count
        packets[variant]=(static,grids)
    assert packets["orig"]==packets[candidate]
    dump(OUT/"classical.json",dict(positions=count,black_static_values=packets["orig"][0],
         both_color_score_grids=len(packets["orig"][1]),mismatches=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="plus")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.candidate, args.out)
