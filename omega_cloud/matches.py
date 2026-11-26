"""One complete game at a time on each four-vCPU runner, with resident engines per rule."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from arena import game, write_game
from datasets import dump, sha
from engine import COORD, Engine, topology

BOOK = ROOT / "data/v3/cloud_openings.json"
FULL = "Omega-Cloud-CPU"
OPPONENT = {"screen":"orig", "extension":"orig", "f15":"Omega-F15-off", "f17":"Omega-F17-off"}
MODELS = {1:"standard15x.onnx", 2:"renju15x.onnx"}
INITIAL, INC = 3_000_000_000, 100_000_000

def schedule(stage):
    book = json.loads(BOOK.read_bytes())
    groups = []
    for rule in (1, 2):
        # Allocate whole mother-opening families to the screen when their sizes sum to 50.
        rows = [r for r in book if r["rule"] == rule]
        rng = random.Random(261126 + rule)
        mothers = sorted({r["mother"] for r in rows})
        rng.shuffle(mothers)
        families = {m:[r for r in rows if r["mother"] == m] for m in mothers}
        for group in families.values():
            rng.shuffle(group)
        subsets = {0:[]}
        for m in mothers:
            for size, family in list(subsets.items()):
                nxt = size + len(families[m])
                if nxt <= 50:
                    subsets.setdefault(nxt, family + [m])
        front = subsets[max(subsets)]
        order = front + [m for m in mothers if m not in front]
        ordered = [r for m in order for r in families[m]]
        assert len(ordered) == len({r["id"] for r in ordered}) == 250
        if stage == "f15":
            if rule == 2:
                return ordered[:100]
        else:
            groups.append(ordered[50:] if stage == "extension" else ordered[:50])
    return [r for pair in zip(*groups, strict=True) for r in pair]

class FischerInfo:
    def submit(self, command):
        lines = []
        for line in command.splitlines():
            lines.append(line)
            if line.startswith("INFO time_left "):
                lines.append(f"INFO time_increment {INC//1_000_000}")
        return super().submit("\n".join(lines))

class Original(FischerInfo, Engine):
    pass

class Omega(FischerInfo, Engine):
    def __init__(self, name, cpus):
        self.variant, self.core, self.cpu_pool = name, cpus[0], [cpus[0]]
        self.search_threads, self.handles, self.oracle_reader = 4, {}, None
        self.weights, self.rule = [], None
        self.cpus = cpus
        dist = Path(os.environ["RAPFI_ARENA_BIN"])
        self.oracle_models = {r:sha(dist/model) for r,model in MODELS.items()}
        env = dict(os.environ, OMEGA_BATCH="4", OMEGA_SHIELD="1", OMEGA_F15=str(int(name != "Omega-F15-off")),
                   OMEGA_F17=str(int(name != "Omega-F17-off")))
        self.proc = subprocess.Popen([str(dist/"pbrain-omega-cloud")], cwd=ROOT, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        try:
            self.ps = psutil.Process(self.proc.pid)
            self.inbox = queue.Queue()
            def drain():
                for raw in self.proc.stdout:
                    self.inbox.put((time.perf_counter_ns(), raw.decode("utf-8",errors="replace").rstrip()))
                self.inbox.put((time.perf_counter_ns(), None))
            self.reader = threading.Thread(target=drain,daemon=True)
            self.reader.start()
            lines, _ = self.exchange("START 15",lambda s:s == "OK")
            assert any("ONNX Runtime 1.22.1 CPUExecutionProvider" in s for s in lines), lines
        except BaseException:
            self.close()
            raise

    def select(self, rule):
        if self.rule != rule:
            self.exchange(f"STOP\nINFO rule {4 if rule == 2 else rule}\nRESTART", lambda s:s == "OK")
            self.weights = [f"{MODELS[rule]} sha256={self.oracle_models[rule]}"]
            self.rule = rule

    def repin(self, core):
        assert core == self.core and self.ps.cpu_affinity() == [core]

    def affinity_masks(self):
        children = self.ps.children()
        assert len(children)==1 and Path(children[0].exe()).name=="pbrain-rapfi-plus-v4"
        masks = {}
        for thread in self.ps.threads():
            aff = sorted(os.sched_getaffinity(thread.id))
            role = "mcts" if aff == self.cpus[:1] else "ort"
            assert aff == (self.cpus[:1] if role=="mcts" else self.cpus[1:3]), (thread.id,aff)
            masks[f"{role}/{thread.id}"] = aff
        for thread in children[0].threads():
            aff = sorted(os.sched_getaffinity(thread.id))
            assert aff == self.cpus[3:]
            masks[f"shield/{thread.id}"] = aff
        assert {k.split("/")[0] for k in masks} == {"mcts","ort","shield"}
        return masks

    def think(self,command="YXNBEST 1",measure_cycles=False,timeout=90):
        lines, ns = self.exchange(command,COORD.fullmatch,timeout)
        meta = json.loads(next(s[14:] for s in lines if s.startswith("MESSAGE OMEGA ")))
        x,y = map(int,lines[-1].split(","))
        assert meta["move"] == x + 15*y and meta["shield_core"] == self.cpus[-1]
        assert not meta["shield_lost"]
        return dict(x=x,y=y,wall_ns=ns,nodes=meta["visits"],depth=None,eval=None,draw_offer=False,
                    engine_ms=meta["elapsed_ms"],omega=meta)

def run(stage, shard, out):
    rows = schedule(stage)[shard::20]
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus)==4 and 0 <= shard < 20 and rows
    players = (FULL, OPPONENT[stage])
    for sub in ("games","psq"):
        (out/sub).mkdir(parents=True)
    dist = Path(os.environ["RAPFI_ARENA_BIN"])
    omega_players = [p for p in players if p!="orig"]
    design = dict(stage=stage, players=players, initial_ns=INITIAL, increment_ns=INC,
                  lane_layout="omega-4t", threads_by_player={p:4 for p in players}, thread_num=None,
                  pondering=False, adjudication=False, exclude=[], book=BOOK.name, book_sha256=sha(BOOK),
                  opening_ids=[r["id"] for r in rows], omega_players=omega_players,
                  lanes=[dict(cpus=[cpus[0],cpus[0]],reserved_vcpus=cpus)],
                  models={str(r):sha(dist/model) for r,model in MODELS.items()},
                  build_commit=(dist/"build-commit").read_text().strip(),
                  binaries={p.name:sha(p) for p in sorted(dist.glob("pbrain-*"))},
                  runner=os.environ["RUNNER_NAME"], shard=shard, clock="3000+100", batch=4, intra_op=2)
    dump(out/"design.json",design)
    dump(out/"hardware.json",topology())
    for rule in sorted({r["rule"] for r in rows}):
        with ExitStack() as stack:
            resident = {name:stack.enter_context(Original("orig", cpus[0], cpu_pool=cpus, search_threads=4)
                                                 if name=="orig" else Omega(name,cpus)) for name in players}
            for e in resident.values():
                e.select(rule)
                e.sync(f"INFO max_depth 99\nINFO show_detail 0\nINFO timeout_match {INITIAL//1_000_000}\nINFO timeout_turn 0")
            for row in rows:
                if row["rule"]!=rule:
                    continue
                for leg in (0,1):
                    g = game(row,leg,[cpus[0],cpus[0]],0,INITIAL,INC,resident=resident,
                             players=players,parallel_reset=True)
                    write_game(out,g)
                    print(row["id"],leg,g["winner"] or "draw",g["reason"],len(g["turns"]),flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=OPPONENT)
    parser.add_argument("shard",type=int)
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    run(args.stage,args.shard,args.out)
