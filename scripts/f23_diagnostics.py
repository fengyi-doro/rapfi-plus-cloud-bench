"""Cloud-only replay of F15 certificates and paired fixed-depth measurements."""
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess

import psutil

from arena import adjudicate
from datasets import DATA, dump, sha
from engine import Engine
from f23_build import ARMS, arm_name, filename
from rules import BITS, has_win, targets, valid_history


def lines(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout.splitlines()


def decode(line):
    game, ply, ordinary, *raw = line.split()
    return game, int(ply), bool(int(ordinary)), [list(map(int, s.split(","))) for s in raw]


def check_line(history, steps):
    assert steps and valid_history(history, 2)
    black = sum(BITS[p] for p in history[::2])
    white = sum(BITS[p] for p in history[1::2])
    for i, (gain, block, other, kind, status) in enumerate(steps):
        if i == 0 and len(history) % 2 == 0:
            assert gain == -1 and kind != 0
        else:
            assert 0 <= gain < 225 and not ((black | white) & BITS[gain])
            white |= BITS[gain]
            assert has_win(white, gain, False) == (kind == 0)
        assert (kind != 1) == (i == len(steps) - 1)
        if kind == 0:
            continue
        assert not targets(black, white, True, 4)
        wins = targets(white, black, False, 4)
        assert 0 <= block < 225 and wins & BITS[block]
        if kind == 2:
            assert block != other and 0 <= other < 225 and wins & BITS[other]
        else:
            assert wins == BITS[block]
            reason = adjudicate([black, white], block, 0, 2)
            if kind == 3:
                assert reason == {2: "forbidden_overline", 3: "forbidden_double_four",
                                  4: "forbidden_double_three"}[status], (reason, status)
            else:
                assert kind == 1 and reason is None
                black |= BITS[block]


def certificate_gate(out):
    games = {}
    for row in (DATA / "f23/renju_games.txt").read_text().splitlines():
        _, game, _, _, n, *moves = row.split()
        assert len(moves) == int(n)
        games[game] = list(map(int, moves))
    rows, white_positions = [], []
    for line in lines(["./f23-gate"]):
        game, ply, ordinary, steps = decode(line)
        h = games[game][:ply]
        check_line(h, steps)
        rows.append(dict(game=game, ply=ply, ordinary=ordinary, steps=steps))
        if not ordinary and ply % 2:
            white_positions.append((game, h))
    assert len(rows) == 252 and len(white_positions) == 41
    dump(out / "certificates.json", rows)
    return white_positions


def player(stack, name, core, rule):
    e = stack.enter_context(Engine(name, core))
    e.select(rule)
    e.sync("YXSHOWINFO\nINFO show_detail 2\nINFO timeout_match 100000000\nINFO timeout_turn 2000000")
    return e


def search(e, moves, depth):
    e.clean_position(moves)
    e.sync(f"INFO max_depth {depth}\nINFO time_left 100000000")
    return e.think("YXNBEST 1", timeout=120)


def measure(out, build, white_positions):
    core = min(psutil.Process().cpu_affinity())
    psutil.Process().cpu_affinity([core])
    positions = json.loads((DATA / "positions.json").read_bytes())
    names = list(build["binaries"])
    for name in names:
        if sha(Path("dist") / filename(name)) != build["binaries"][name]:
            raise ValueError(f"diagnostic binary mismatch: {name}")
    rows, chosen = [], []
    for rule in (1, 2):
        roots = [r for r in positions if r["rule"] == rule and 10 <= len(r["moves"]) <= 80][:16]
        assert len(roots) == 16
        with ExitStack() as stack:
            players = {name: player(stack, name, core, rule) for name in names}
            for i, pos in enumerate(roots):
                for rep in (0, 1):
                    order = names[i % len(names):] + names[:i % len(names)]
                    for name in (order if rep == 0 else reversed(order)):
                        t = search(players[name], pos["moves"], 7)
                        if t["nodes"] is None:
                            raise ValueError(f"fixed-depth node counter missing on {name}")
                        rows.append(dict(rule=rule, position=pos["id"], repeat=rep, player=name, **t))
            if rule == 2:
                for game, history in white_positions:
                    tactics = [(4, "orig"), (4, "v4"), (4, arm_name(1)), (8, "v4"), (8, arm_name(2))]
                    tactics += [(8, label) for label in (arm_name(18), arm_name(30)) if label in players]
                    for depth, arm in tactics:
                        if arm not in players:
                            continue
                        t = search(players[arm], history, depth)
                        move = t["x"] + 15 * t["y"]
                        chosen.append(dict(game=game, ply=len(history), depth_cap=depth, player=arm,
                                           move=move, history=history, **t))
    queries = "".join(" ".join(map(str, (1, r["move"], len(r["history"]), *r["history"]))) + "\n" for r in chosen)
    for r, line in zip(chosen, lines(["./f23-gate", "chosen"], input=queries), strict=True):
        _, _, _, steps = decode(line)
        h = r.pop("history")
        if steps:
            assert steps[0][0] == r["move"]
            check_line(h, steps)
        r["chosen_certified"] = bool(steps)
    costs = {}
    for name in names:
        subset = [r for r in rows if r["player"] == name]
        total_ns = sum(r["wall_ns"] for r in subset)
        costs[name] = dict(observations=len(subset), nodes=sum(r["nodes"] for r in subset),
                           wall_ms=total_ns / 1e6, mean_ms=total_ns / 1e6 / len(subset),
                           completed_depth7=sum(r["depth"] >= 7 for r in subset))
    tactics = {}
    for (depth, name), n in Counter((r["depth_cap"], r["player"]) for r in chosen).items():
        sample = [r for r in chosen if (r["depth_cap"], r["player"]) == (depth, name)]
        tactics[f"{name}-d{depth}"] = dict(positions=n, certified=sum(r["chosen_certified"] for r in sample),
                                           mate_reports=sum(str(r.get("eval", "")).lstrip("+").startswith(("M", "W"))
                                                            or (str(r.get("eval", "")).lstrip("+").isdigit()
                                                                and int(r["eval"]) >= 29500) for r in sample))
    dump(out / "fixed_depth_rows.json", rows)
    dump(out / "tactical_rows.json", chosen)
    dump(out / "diagnostics.json", dict(renju_positions=9069, ordinary=172, augmented=252,
                                        additional_white_to_move=41, additional_black_to_move=39,
                                        native_replay_mismatches=0, independent_certificate_mismatches=0,
                                        cpu=core, depth_cap=7, positions=32, repeats=2, costs=costs, tactics=tactics))
    print(json.dumps(dict(costs=costs, tactics=tactics)))


if __name__ == "__main__":
    os.environ.update(RAPFI_ARENA_BIN="dist", RAPFI_ROOT_MODE="control")
    output = Path("diagnostics")
    output.mkdir(exist_ok=True)
    measure(output, json.loads(Path("dist/f23-build.json").read_bytes()), certificate_gate(output))
