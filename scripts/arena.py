"""Paired Gomocup games. Fresh processes, physical-core swaps and external Fischer clocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import random
import threading
import time
import psutil
from datasets import DATA, ROOT, SEED, dump, sha
from engine import DeadlineExpired, Engine, filename, topology, utc
from bit_gate import board_command
from rules import BITS, FIVES, SIXES, black_status, exact, has_win, valid_history

INITIAL = 3_000_000_000
INCREMENT = 80_000_000
PRINT = threading.Lock()


def adjudicate(stones, p, color, rule):
    own = stones[color] | BITS[p]
    opp = stones[1-color]
    if has_win(own, p, exact(rule,color)):
        over = any(own & m == m for m,_,_ in SIXES[p])
        return "five_or_more" if rule == 2 and color == 1 and over else "five"
    if rule == 2 and color == 0 and black_status(own, opp, p) < 0:
        if any(own & m == m for m,_,_ in SIXES[p]):
            return "forbidden_overline"
        fours = {own & m for m, ends, _ in FIVES[p] if not opp & m and not own & ends
                 and (own & m).bit_count() == 4}
        return "forbidden_double_four" if len(fours) >= 2 else "forbidden_double_three"
    return None


def game(opening, leg, cores, lane, initial=INITIAL, increment=INCREMENT, resident=None, boot_epoch=None,
         players=("plus","orig"), early_result=None, parallel_reset=False):
    first_plus_black = int(opening["id"][1:]) % 2 == 1
    plus_black = first_plus_black != bool(leg)
    names = players if plus_black else players[::-1]
    rule = opening["rule"]
    history = opening["moves"].copy()
    assert valid_history(history, rule)
    stones = [sum(BITS[p] for p in history[::2]), sum(BITS[p] for p in history[1::2])]
    bank = [initial, initial]
    turns = []
    started = utc()
    started_qpc = time.perf_counter_ns()
    winner, reason = None, "full_board"
    evidence = None
    with ExitStack() as stack:
        if resident is None:
            engines = [stack.enter_context(Engine(name, core)) for name, core in zip(names,cores,strict=True)]
        else:
            engines = [resident[name] for name in names]
            for engine,core in zip(engines,cores,strict=True):
                engine.repin(core)
        for engine in engines:
            engine.select(rule)
        if parallel_reset:
            command="STOP\nRESTART\nYXHASHCLEAR\n"+board_command(history)+f"\nINFO time_left {initial//1_000_000}\nABOUT"
            stamps=[e.submit(command) for e in engines]
            for e,stamp in zip(engines,stamps):
                e.collect(stamp,command,lambda s:"author=" in s)
        else:
            for engine in engines:
                engine.sync(f"INFO max_depth 99\nINFO show_detail 0\nINFO timeout_match {initial//1_000_000}\nINFO timeout_turn 0")
                engine.clean_position(history)
        # The previous turn fixes this player's next bank. Acknowledge its INFO
        # while the opponent is thinking. The move clock starts only after ABOUT.
        def announce(e,ns):
            cmd=f"INFO time_left {ns//1_000_000}\nABOUT"
            return e.submit(cmd),cmd
        pending=({e:announce(e,initial) for e in engines} if resident is not None and not parallel_reset else {})
        pids = [e.proc.pid for e in engines]
        first_affinity = {n:e.affinity_masks() for n,e in zip(names,engines,strict=True)}
        while len(history) < 225:
            color = len(history) % 2
            e = engines[color]
            # The official clock measures command write through complete coordinate receipt.
            # Option changes and STOP/ABOUT synchronization precede that interval.
            left_ms = max(0, bank[color] // 1_000_000)
            if resident is None:
                e.sync(f"STOP\nINFO time_left {left_ms}")
            elif e in pending:
                stamp,cmd=pending.pop(e)
                assert cmd.startswith(f"INFO time_left {left_ms}\n")
                e.collect(stamp,cmd,lambda s:"author=" in s)
            else:
                assert parallel_reset and bank[color]==initial and len(turns)<2
            command = "YXNBEST 1" if not turns else f"TURN {history[-1]%15},{history[-1]//15}"
            try:
                result = e.think(command, timeout=bank[color]/1e9)
            except DeadlineExpired as deadline:
                result = dict(x=None, y=None, wall_ns=deadline.elapsed_ns, nodes=None, depth=None,
                              eval=None, engine_ms=None, no_coordinate=True)
            before = bank[color]
            result.update(engine=names[color], color=color, before_ns=before, info_time_left_ms=left_ms)
            turns.append(result)
            if result.get("no_coordinate") or result["wall_ns"] > before:
                result["after_ns"] = 0
                winner, reason = names[1-color], "time"
                if resident is not None:
                    e.sync("STOP")
                break
            bank[color] = before - result["wall_ns"] + increment
            result["after_ns"] = bank[color]
            x, y = result["x"], result["y"]
            if not (0 <= x < 15 and 0 <= y < 15) or (stones[0] | stones[1]) & BITS[y*15+x]:
                winner, reason = names[1-color], "illegal_coordinate"
                break
            p = y*15+x
            verdict = adjudicate(stones,p,color,rule)
            history.append(p)
            stones[color] |= BITS[p]
            if verdict:
                winner = names[1-color] if verdict.startswith("forbidden") else names[color]
                reason = verdict
                break
            if early_result is not None and len(history)<225:
                decision = early_result(stones, color, rule, turns, len(history))
                if decision:
                    winning_color, reason, evidence = decision
                    winner = names[winning_color] if winning_color is not None else None
                    break
            if resident is not None and len(history)<225:
                pending[e]=announce(e,bank[color])
        for e,(stamp,cmd) in pending.items():
            e.collect(stamp,cmd,lambda s:"author=" in s)
        assert all(e.ps.cpu_affinity() == [core] for e,core in zip(engines,cores,strict=True))
        final_affinity = {n:e.affinity_masks() for n,e in zip(names,engines,strict=True)}
        weights = {n:list(dict.fromkeys(e.weights)) for n,e in zip(names,engines,strict=True)}
    return dict(opening=opening["id"], leg=leg, rule=rule, black=names[0], white=names[1],
                lane=lane, cores=dict(zip(names,cores)), pids=dict(zip(names,pids)), started=started, ended=utc(),
                started_qpc_ns=started_qpc, ended_qpc_ns=time.perf_counter_ns(),
                boot_time=int(psutil.boot_time()) if boot_epoch is None else boot_epoch,
                initial_ns=initial, increment_ns=increment, opening_plies=len(opening["moves"]),
                moves=history, turns=turns, winner=winner, reason=reason, weights=weights,
                affinity_before=first_affinity, affinity_after=final_affinity,
                **({"adjudication": evidence} if early_result is not None else {}))


def write_game(directory, result):
    stem = f"{result['opening']}_{result['leg']}"
    target = directory/"games"/f"{stem}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",",":")), encoding="utf-8")
    temporary.replace(target)
    write_psq(directory,result)


def write_psq(directory,result):
    stem = f"{result['opening']}_{result['leg']}"
    milliseconds = [0]*result["opening_plies"] + [t["wall_ns"]//1_000_000 for t in result["turns"]]
    psq = ["Piskvorky 15x15, 0:0, 0"]
    psq += [f"{p%15+1},{p//15+1},{ms}" for p,ms in zip(result["moves"],milliseconds)]
    psq += [f"{result['black']} - {result['white']}", f"{result['winner'] or 'draw'} ({result['reason']})"]
    (directory/"psq"/f"{stem}.psq").write_text("\n".join(psq)+"\n",encoding="utf-8")


def run(directory, v2=False, benchmark_dir=None):
    directory.mkdir(parents=True, exist_ok=True)
    for folder in ("games", "psq"):
        (directory/folder).mkdir(exist_ok=True)
    hashes = {v:sha(filename(v)) for v in ("orig","plus")}
    bench=benchmark_dir.resolve() if benchmark_dir else ROOT/("bench/v2/final" if v2 else "bench/final")
    for path in (ROOT/"validation/heads.json", ROOT/"validation/tactics.json", bench/"summary.json"):
        result = json.loads(path.read_bytes())
        assert all(result["exe"][v] == hashes[v] for v in hashes), f"acceptance binary changed: {path}"
    book=DATA/("v2/openings.json" if v2 else "openings.json")
    openings = json.loads(book.read_bytes())
    assert [sum(r["rule"] == rule for r in openings) for rule in (1,2)] == ([150,150] if v2 else [60,40])
    initial,increment=(1_500_000_000,40_000_000) if v2 else (INITIAL,INCREMENT)
    random.Random(SEED).shuffle(openings)
    current = topology()
    hardware_path=directory/"hardware.json"
    hardware = json.loads(hardware_path.read_bytes()) if hardware_path.exists() else current
    assert hardware["cores"] == current["cores"], "CPU topology changed since the first game"
    lanes = min(3 if v2 else 4, len(hardware["performance"])//2)
    assert lanes >= 1
    perf = hardware["performance"][:2*lanes]
    if hardware["helper"]:
        psutil.Process().cpu_affinity(hardware["helper"])
    design = dict(exe=hashes, book_sha256=sha(book), initial_ns=initial,
                  increment_ns=increment, lanes=lanes, seed=SEED, thread_num=1, hash_kib=8192,
                  node_cap=0, pondering=False, show_detail=0, core_order="alternate black core between openings",
                  restart=("resident engines + RESTART + YXHASHCLEAR" if v2 else "fresh process + RESTART + YXHASHCLEAR"))
    if v2:
        design.update(stage=2,paired_games=600,opening_path="data/v2/openings.json",
                      score_plan="600 fixed games; rule-stratified opening-pair bootstrap, and a mother-cluster sensitivity analysis")
        if benchmark_dir:
            design["benchmark_path"]=(bench.relative_to(ROOT) if bench.is_relative_to(ROOT) else bench).as_posix()
    design_path = directory/"design.json"
    if design_path.exists():
        assert json.loads(design_path.read_bytes()) == design, "existing match uses a different design"
    else:
        dump(design_path,design)
    expected=[directory/"games"/f"{o['id']}_{leg}.json" for o in openings for leg in (0,1)]
    if all(p.exists() for p in expected):
        for path in expected:
            if not (directory/"psq"/path.with_suffix(".psq").name).exists():
                write_psq(directory,json.loads(path.read_bytes()))
        if not (directory/"timing.json").exists():
            saved=[json.loads(p.read_bytes()) for p in expected]
            dump(directory/"timing.json",dict(start=min(g["started"] for g in saved),
                 end=max(g["ended"] for g in saved),count=len(saved)))
        print("All scheduled games already archived",len(expected),flush=True)
        return
    start = utc()
    start_qpc=time.perf_counter_ns()
    boot_epoch=int(psutil.boot_time())
    if not hardware_path.exists():
        dump(hardware_path,hardware)
    with (directory/"sessions.jsonl").open("a",encoding="utf-8") as log:
        log.write(json.dumps(dict(start=start,start_qpc_ns=start_qpc,boot_time=boot_epoch,topology=current))+"\n")
    monitor_end = threading.Event()
    def sample_load():
        with (directory/"host_load.jsonl").open("a",encoding="utf-8") as log:
            psutil.cpu_percent(percpu=True)
            while not monitor_end.wait(2):
                log.write(json.dumps(dict(utc=utc(),cpu_pct=psutil.cpu_percent(percpu=True)))+"\n")
    monitor = threading.Thread(target=sample_load,daemon=True)
    monitor.start()
    def lane_work(lane):
        lane_cores = perf[2*lane:2*lane+2]
        with ExitStack() as processes:
            resident = ({v:processes.enter_context(Engine(v,core)) for v,core in zip(("orig","plus"),lane_cores)}
                        if v2 else None)
            play_lane(lane,lane_cores,resident)
    def play_lane(lane,lane_cores,resident):
        for index,opening in enumerate(openings[lane::lanes]):
            slots = lane_cores if (index+lane)%2 == 0 else lane_cores[::-1]
            for leg in (0,1):
                path = directory/"games"/f"{opening['id']}_{leg}.json"
                if path.exists():
                    if not (directory/"psq"/path.with_suffix(".psq").name).exists():
                        write_psq(directory,json.loads(path.read_bytes()))
                    continue
                result = game(opening,leg,slots,lane,initial,increment,resident,boot_epoch)
                write_game(directory,result)
                with PRINT:
                    print(opening["id"],leg,result["winner"] or "draw",result["reason"],len(result["moves"]),flush=True)
    try:
        with ThreadPoolExecutor(lanes) as pool:
            list(pool.map(lane_work,range(lanes)))
    finally:
        monitor_end.set()
        monitor.join()
    first=json.loads((directory/"sessions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    elapsed=(time.perf_counter_ns()-first["start_qpc_ns"] if "start_qpc_ns" in first
             and first["boot_time"]==boot_epoch else None)
    dump(directory/"timing.json", dict(start=first["start"],end=utc(),wall_qpc_ns=elapsed,
                                        count=len(list((directory/"games").glob("*.json")))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"match")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--benchmark-dir",type=Path,help="use a separately measured fixed-depth benchmark for this season")
    args=parser.parse_args()
    run(args.out,args.v2,args.benchmark_dir)
