"""Recover colour-specific outcomes and win counts per opening from verified match artifacts."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from datasets import dump


def summarize():
    verified = json.loads(Path("summary/summary.json").read_bytes())
    challenger, original = verified["players"]
    expected = {r["opening"]:r["half_points"] for r in verified["pair_ledger"]}
    if verified["binary_hashes"]["Linux"][original] != "622200dd31b39e4120cccec3ea3a47eaadda8c7fc231e7763b514474fea26ab9":
        raise ValueError("match does not have the frozen original binary")
    outcomes = defaultdict(Counter)
    flags, illegal = Counter(), Counter()
    pairs = defaultdict(dict)
    for path in sorted(Path("shards").glob("shard-*/games/*.json")):
        game = json.loads(path.read_bytes())
        if {game["black"], game["white"]} != {challenger, original}:
            raise ValueError(f"unexpected players in {path}")
        rule = {1:"Standard", 2:"Renju"}[game["rule"]]
        color = "black" if game["black"] == challenger else "white"
        result = {challenger:"W", original:"L", None:"D"}[game["winner"]]
        opening, leg = game["opening"], game["leg"]
        if opening not in expected or leg not in (0, 1) or leg in pairs[opening]:
            raise ValueError(f"invalid opening or leg in {path}")
        pairs[opening][leg] = (color, result)
        outcomes[rule, color][result] += 1
        outcomes["All", color][result] += 1
        if game["reason"] == "time":
            flags[game["turns"][-1]["engine"]] += 1
        if game["reason"] == "illegal_coordinate" or game["reason"].startswith("forbidden"):
            illegal[game["turns"][-1]["engine"]] += 1
    if expected.keys() != pairs.keys():
        raise ValueError("archived opening coverage differs from the verified match")
    wins = {}
    for opening, legs in pairs.items():
        if {s[0] for s in legs.values()} != {"black", "white"}:
            raise ValueError(f"colour pair incomplete for {opening}")
        points = sum({"W":2,"D":1,"L":0}[r] for _, r in legs.values())
        if points != expected[opening]:
            raise ValueError(f"archived game points differ for {opening}")
        wins[opening] = sum(r == "W" for _, r in legs.values())
    for rule, r in verified["scores"].items():
        total = outcomes[rule, "black"] + outcomes[rule, "white"]
        if any(total[k] != r[k] for k in ("W","D","L")):
            raise ValueError(f"colour totals do not match verified {rule} outcomes")
    audit = [s["audit"] for s in verified["shards"]]
    if sum(flags.values()) != sum(r["hard_flags"] for r in audit) or sum(illegal.values()) != sum(r["illegal_moves"] for r in audit):
        raise ValueError("terminal events do not match the independent replay")
    directory = Path("outcomes")
    directory.mkdir(exist_ok=True)
    dump(directory/"outcomes.json", dict(
        match_run=int(os.environ["MATCH_RUN"]), commit=verified["protocol"]["commit"], challenger=challenger,
        original_sha256=verified["binary_hashes"]["Linux"][original], book_sha256=verified["protocol"]["book_sha256"],
        run=f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
        colour={r:{c:{k:outcomes[r,c][k] for k in ("W","D","L")} for c in ("black","white")}
                for r in ("All","Standard","Renju")},
        pair_wins=wins, flags_by_player=dict(flags), illegal_by_player=dict(illegal)))


if __name__ == "__main__":
    summarize()
