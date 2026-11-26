"""Paired Gomocup arena with disjoint homogeneous core pairs and independently checkable verdicts."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import time

import psutil

from arena import game, write_game, write_psq
from arena_certificates import forced_line, standard_pairs
from datasets import DATA, ROOT, SEED, dump, sha
from engine import Engine, filename, topology, utc
from rules import BITS, BOARD, ID, STARTS, exact, targets


def core_groups(hardware, excluded, maximum):
    allowed = set(psutil.Process().cpu_affinity())
    groups = defaultdict(dict)
    hybrid=len({c["efficiency"] for c in hardware["cores"]})>1
    for c in hardware["cores"]:
        lp = c["logical"]
        if lp in excluded or lp not in allowed or (os.name == "nt" and c["group"]):
            continue
        # On Windows hybrid parts a separate LLC domain can contain low-power
        # E cores sharing the ordinary E cores' efficiency class.
        domain=c.get("last_level_cache",0) if hybrid else 0
        groups[(c["efficiency"],c["group"],domain)].setdefault((c["group"],c["physical"]),lp)
    lanes = []
    for (cls,_,_), rows in sorted(groups.items(),key=lambda item:(-item[0][0],item[0][1:])):
        cores = sorted(rows.values())
        for i in range(0, len(cores)-1, 2):
            lanes.append(dict(efficiency=cls, cpus=cores[i:i+2]))
    if not lanes:
        raise ValueError("two available homogeneous physical cores are required")
    return lanes[:maximum]


def no_open_line(stones):
    # Every length-five segment contains both colours. Future black Renju
    # replies also stay legal: even an unblocked five or overline is impossible.
    for opp in stones:
        available = BOARD ^ opp
        for shift, starts in STARTS.items():
            possible = starts
            for i in range(5):
                possible &= available >> (i*shift)
            if possible: return False
    return True


def early_result(stones, last, rule, turns, ply):
    # A move completing an exact five has priority over every Renju foul.
    next_side = 1-last
    next_wins = targets(stones[next_side], stones[last], exact(rule,next_side), 4)
    if next_wins:
        p = ID[next_wins & -next_wins]
        return next_side, "forced_five", dict(points=[p], color=next_side)
    threats = targets(stones[last], stones[next_side], exact(rule,last), 4)
    if threats.bit_count() >= 2:
        first = threats & -threats
        other = threats ^ first
        return last, "forced_double_five", dict(points=[ID[first], ID[other & -other]], color=last)
    # A player's offered draw is valid through the opponent's next reply only.
    if len(turns) >= 2 and all(t.get("draw_offer") for t in turns[-2:]):
        return None, "mutual_draw_offer", dict(plies=[ply-1,ply], message="MESSAGE DRAW_OFFER")
    # The arena's bilateral zero-score agreement requires two protocol zeros
    # from each participant and an independent structural draw certificate.
    if (ply >= 120 and len(turns) >= 4 and all(t["eval"] == "0" for t in turns[-4:])
            and no_open_line(stones)):
        return None, "mutual_certified_draw", dict(scores=["0"]*4, plies=list(range(ply-3,ply+1)),
                                                    certificate="every five contains both colours")
    return None


def certified_result(stones, last, rule, turns, ply):
    verdict=early_result(stones,last,rule,turns,ply)
    if verdict: return verdict
    line=forced_line(stones,last,rule,turns[-1])
    if line: return last,"forced_vcf",dict(color=last,steps=line)
    if (rule==1 and ply>=120 and ply%4==0 and len(turns)>=4
            and all(t["eval"]=="0" for t in turns[-4:])):
        certificate=standard_pairs(stones,1-last)
        if certificate is not None:
            return None,"mutual_paired_draw",dict(scores=["0"]*4,pairs=certificate)
    return None


def load_book(path, pairs, shard, shards):
    rows = json.loads(path.read_bytes())
    groups = {r:[o for o in rows if o["rule"] == r] for r in (1,2)}
    ordered = [o for pair in zip(groups[1], groups[2], strict=True) for o in pair]
    if not 1 <= shards <= pairs <= len(ordered):
        raise ValueError("each shard requires at least one pair from the book")
    if len({o["id"] for o in ordered}) != len(ordered) or not 1 <= shard <= shards:
        raise ValueError("invalid opening IDs or shard index")
    count, extra = divmod(pairs, shards)
    start = (shard - 1) * count + min(shard - 1, extra)
    return ordered[start:start + count + (shard <= extra)]


def play_lane(lane, cores, helper, openings, directory, design, work_queue=None):
    if design.get("one_vcpu_per_game"):
        psutil.Process().cpu_affinity([cores[0]])
    elif helper:
        psutil.Process().cpu_affinity(helper)
    initial, increment = design["initial_ns"], design["increment_ns"]
    names = tuple(design["players"])
    fast=design.get("parallel_reset",False)
    start_order={o["id"]:i for i,o in enumerate(openings)}
    schedule_order={k:i for i,k in enumerate(design["opening_ids"])}
    primary=set(design["opening_ids"][:200])
    def work():
        yield from sorted(openings,key=lambda o:(o["id"] not in primary,o["rule"]))
        if work_queue is not None:
            while (o:=work_queue.get()) is not None:
                yield o
    with ExitStack() as stack:
        engines = {v:stack.enter_context(Engine(v, c)) for v,c in zip(names, cores, strict=True)}
        if fast:
            for e in engines.values():
                e.sync(f"INFO max_depth 99\nINFO show_detail 0\nINFO timeout_match {initial//1_000_000}\nINFO timeout_turn 0")
        for o in work():
            index=(schedule_order if work_queue is not None else start_order)[o["id"]]
            slots = cores if (index+lane) % 2 == 0 else cores[::-1]
            for leg in (0,1):
                path = directory/"games"/f"{o['id']}_{leg}.json"
                if path.exists():
                    prior = json.loads(path.read_bytes())
                    if not (directory/"psq"/f"{o['id']}_{leg}.psq").exists():
                        write_psq(directory,prior)
                    continue
                result = game(o,leg,slots,lane,initial,increment,engines,design["boot_time"],
                              names, ((certified_result if fast else early_result) if design["adjudication"] else None),
                              parallel_reset=fast)
                write_game(directory,result)
    return lane


@contextmanager
def exclusive_output(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"arena.lock").open("a+b") as stream:
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another arena is already writing {directory}") from exc
        yield


def run(args):
    with exclusive_output(args.out):
        run_locked(args)


def run_locked(args):
    out, book = args.out, args.book
    if not 1 <= args.lanes <= 5 or min(args.initial_ms, args.increment_ms) < 0 or args.initial_ms == 0:
        raise ValueError("invalid lane count or time control")
    if args.challenger == args.opponent:
        raise ValueError("two distinct variants are required")
    for f in ("games","psq"):
        (out/f).mkdir(parents=True, exist_ok=True)
    openings = load_book(book, args.pairs, args.shard, args.shards)
    hardware = topology(core_domains=True)
    excluded = set(args.exclude)
    limit = min(args.lanes, len(openings))
    if args.vcpu_lanes:
        required = {"avx2", "bmi2", "fma"}
        missing = required.difference(hardware.get("cpu_flags", []))
        if missing:
            raise ValueError(f"runner lacks required CPU features: {sorted(missing)}")
        available = sorted(set(psutil.Process().cpu_affinity()) - excluded)
        if len(available) < limit:
            raise ValueError(f"requested {limit} vCPU lanes, runner exposes {len(available)}")
        # With pondering disabled, black and white alternate searches on the same vCPU.
        lanes = [dict(cpus=[cpu, cpu]) for cpu in available[:limit]]
    else:
        lanes = core_groups(hardware, excluded, limit)
    used = {cpu for lane in lanes for cpu in lane["cpus"]}
    helper = sorted(set(psutil.Process().cpu_affinity()) - used - excluded)
    names = [args.challenger, args.opponent]
    design = dict(players=names, exe={v:sha(filename(v)) for v in names}, commit=os.environ.get("GITHUB_SHA"),
                  book_sha256=sha(book), book=book.resolve().name,
                  opening_ids=[o["id"] for o in openings], lanes=lanes, exclude=sorted(excluded),
                  initial_ns=args.initial_ms*1000000, increment_ns=args.increment_ms*1000000,
                  shard=args.shard, shards=args.shards, total_pairs=args.pairs,
                  adjudication=not args.no_adjudication, boot_time=int(psutil.boot_time()),
                  seed=SEED, thread_num=1, hash_kib=8192, pondering=False,
                  runtime=json.loads((DATA/"runtime_files.json").read_bytes()))
    if not args.basic:
        scheduler = "next available vCPU" if args.vcpu_lanes else "next available homogeneous pair"
        design.update(scheduler=scheduler,parallel_reset=True)
    if args.vcpu_lanes:
        design["one_vcpu_per_game"] = True
        design["build_profile"] = os.environ.get("RAPFI_BUILD_PROFILE")
    if any(sha(ROOT/p) != h for p,h in design["runtime"].items()):
        raise ValueError("weight/config checksum mismatch")
    if (out/"design.json").exists():
        prior = json.loads((out/"design.json").read_bytes())
        prior.setdefault("commit",None)
        if {k:v for k,v in prior.items() if k!="boot_time"} != {k:v for k,v in design.items() if k!="boot_time"}:
            raise ValueError("resume requires the same match design")
    else:
        dump(out/"design.json",design)
    expected = {f"{o['id']}_{leg}" for o in openings for leg in (0,1)}
    existing = {p.stem for p in (out/"games").glob("*.json")}
    if existing - expected:
        raise ValueError("output contains unexpected games")
    if existing == expected and (out/"timing.json").exists():
        for stem in existing:
            if not (out/"psq"/f"{stem}.psq").exists():
                write_psq(out,json.loads((out/"games"/f"{stem}.json").read_bytes()))
        return
    start, tick = utc(), time.perf_counter()
    sessions=out/"sessions.jsonl"
    with sessions.open("a",encoding="utf8") as log:
        log.write(json.dumps(dict(start=start,resumed_games=len(existing)))+"\n")
    dump(out/"hardware.json",hardware)
    print("cores",lanes,"excluded",sorted(excluded),"pairs",len(openings),flush=True)
    with ExitStack() as processes:
        controller=psutil.Process()
        previous_affinity=controller.cpu_affinity()
        controller.cpu_affinity(helper or sorted(used))
        processes.callback(controller.cpu_affinity,previous_affinity)
        queue=None
        if args.basic:
            assignments=[openings[i::len(lanes)] for i in range(len(lanes))]
        else:
            manager=processes.enter_context(multiprocessing.Manager())
            queue=manager.Queue()
            assignments=[[] for _ in lanes]
            primary=set(design["opening_ids"][:200])
            for o in sorted(openings,key=lambda row:(row["id"] not in primary,row["rule"])):
                legs=[out/"games"/f"{o['id']}_{leg}.json" for leg in (0,1)]
                present=[file for file in legs if file.exists()]
                if present:
                    for file in present:
                        if not (out/"psq"/file.with_suffix(".psq").name).exists():
                            write_psq(out,json.loads(file.read_bytes()))
                    if len(present)==1:
                        owner=json.loads(present[0].read_bytes())["lane"]
                        assignments[owner].append(o)
                else:
                    queue.put(o)
            for _ in lanes: queue.put(None)
        pool=processes.enter_context(ProcessPoolExecutor(len(lanes)))
        futures=[pool.submit(play_lane,i,lane["cpus"],helper,assignments[i],out,design,queue)
                 for i,lane in enumerate(lanes)]
        for f in futures: f.result()
    timing = dict(start=start,end=utc(),wall_seconds=time.perf_counter()-tick,
                  count=len(list((out/"games").glob("*.json"))),resumed_games=len(existing))
    if timing["count"] != len(expected): raise RuntimeError("incomplete schedule")
    if existing:
        first=json.loads(sessions.read_text(encoding="utf8").splitlines()[0])["start"]
        earliest=min(json.loads((out/"games"/f"{stem}.json").read_bytes())["started"] for stem in existing)
        timing["session_seconds"]=timing["wall_seconds"]
        timing["start"]=min(first,earliest)
        timing["wall_seconds"]=(datetime.fromisoformat(timing["end"])-datetime.fromisoformat(timing["start"])).total_seconds()
    dump(out/"timing.json",timing)
    print(timing, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--book",type=Path,default=DATA/"v2/openings.json")
    p.add_argument("--pairs",type=int,default=200,help="total opening pairs before partitioning")
    p.add_argument("--shard",type=int,default=1)
    p.add_argument("--shards",type=int,default=1)
    p.add_argument("--lanes",type=int,default=5)
    p.add_argument("--vcpu-lanes",action="store_true",help="alternate both single-threaded engines on one dedicated Linux vCPU per game")
    p.add_argument("--exclude",type=int,nargs="*",default=[14,15] if os.name=="nt" else [])
    players = ["v4","v3","v2","orig"]
    if os.environ.get("RAPFI_ARENA_BIN") and os.environ.get("RAPFI_CANDIDATE_LABEL"):
        players.append(os.environ["RAPFI_CANDIDATE_LABEL"])
    p.add_argument("--challenger",choices=players,default="v3")
    p.add_argument("--opponent",choices=["v4","v3","v2","orig"],default="orig")
    p.add_argument("--initial-ms",type=int,default=1500)
    p.add_argument("--increment-ms",type=int,default=40)
    p.add_argument("--no-adjudication",action="store_true")
    p.add_argument("--basic",action="store_true",help="reproduce static lanes and the immediate-win/draw referee")
    run(p.parse_args())

