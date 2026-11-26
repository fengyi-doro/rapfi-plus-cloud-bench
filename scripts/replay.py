"""Second, coordinate-based referee. Replays both clocks and all archived games."""
import argparse
import json
from pathlib import Path
from datasets import DATA, ROOT, dump, sha
from arena import INITIAL, INCREMENT, adjudicate
from rules import BITS, bits, replay, replies_in_tree

DIRECTIONS = ((1,0),(0,1),(1,1),(1,-1))


def cell(x, y):
    return (x,y) if 0 <= x < 15 and 0 <= y < 15 else None


def lines_through(grid, pos, color):
    lengths = []
    for dx,dy in DIRECTIONS:
        length = 1
        for sign in (1,-1):
            x,y = pos
            while True:
                x,y = x+sign*dx, y+sign*dy
                if grid.get((x,y)) != color:
                    break
                length += 1
        lengths.append(length)
    return lengths


def verdict(grid, pos, color, rule):
    lengths = lines_through(grid,pos,color)
    if 5 in lengths or (rule == 0 or (rule == 2 and color == 1)) and max(lengths) >= 5:
        return "five_or_more" if rule == 2 and color == 1 and max(lengths) >= 6 else "five"
    if color != 0 or rule != 2:
        return None
    if max(lengths) > 5:
        return "forbidden_overline"
    fours, threes = set(), {}
    for dx,dy in DIRECTIONS:
        x,y = pos
        at = lambda k:cell(x+dx*k,y+dy*k)
        for length in (4,5):
            for start in range(1-length,1):
                window = [at(k) for k in range(start,start+length)]
                if None in window or any(grid.get(p) == 1 for p in window):
                    continue
                empty = [p for p in window if p not in grid]
                if len(empty) != 1:
                    continue
                stones = frozenset(p for p in window if p in grid)
                left, right = at(start-1), at(start+length)
                if length == 5:
                    if grid.get(left) != 0 and grid.get(right) != 0:
                        fours.add(stones)
                elif left and right and left not in grid and right not in grid:
                    if grid.get(at(start-2)) != 0 and grid.get(at(start+length+1)) != 0:
                        threes.setdefault(stones,set()).add(empty[0])
    if len(fours) >= 2:
        return "forbidden_double_four"
    if len(threes) >= 2:
        live = 0
        for extensions in threes.values():
            for ext in extensions:
                grid[ext] = 0
                value = verdict(grid,ext,0,2)
                del grid[ext]
                if value is None:
                    live += 1
                    break
            if live >= 2:
                return "forbidden_double_three"
    return None


def referee_fixtures():
    xy=(7,7)
    horizontal=lambda xs:{(x,7) for x in xs}
    vertical=lambda ys:{(7,y) for y in ys}
    both=horizontal([6,8]) | vertical([6,8])
    templates=[
        ("standard_five",1,0,horizontal(range(3,7)),"five"),
        ("standard_overline",1,0,horizontal([4,5,6,8,9]),None),
        ("white_overline",2,1,horizontal([4,5,6,8,9]),"five_or_more"),
        ("black_overline",2,0,horizontal([4,5,6,8,9]),"forbidden_overline"),
        ("exact_five_priority",2,0,horizontal(range(3,7))|vertical(range(2,7)),"five"),
        ("two_threes",2,0,both,"forbidden_double_three"),
        ("two_fours",2,0,horizontal([4,5,6])|vertical([4,5,6]),"forbidden_double_four"),
        ("two_collinear_fours",2,0,horizontal([3,5,6,9]),"forbidden_double_four"),
        ("two_ends_of_one_four",2,0,horizontal([4,5,6]),None)]
    result=[]
    def both_oracles(name, grid, pos, side, rule, expected):
        stones=[0,0]
        for (x,y),color in grid.items():
            stones[color] |= BITS[15*y+x]
        label=adjudicate(stones,pos[1]*15+pos[0],side,rule)
        grid[pos]=side
        assert verdict(grid,pos,side,rule)==label==expected,name
        result.append(name)
    for name,rule,side,pieces,expected in templates:
        both_oracles(name,{p:side for p in pieces},xy,side,rule,expected)
    fixture=json.loads((DATA/"recursive_foul.json").read_bytes())
    grid={(p%15,p//15):i%2 for i,p in enumerate(fixture["moves"])}
    a=fixture["black_attack"]
    target=(a%15,a//15)
    both_oracles("recursive_false_three",grid.copy(),target,0,2,None)
    w=fixture["extra_white"]
    grid[(w%15,w//15)]=1
    both_oracles("recursive_true_three",grid,target,0,2,"forbidden_double_three")
    return result


def early_verdict(grid, game, names):
    reason, evidence = game["reason"], game["adjudication"]
    next_side = len(grid)%2
    if reason == "forced_vcf":
        last=1-next_side
        assert game["turns"][-1]["eval"].startswith("+M") and evidence["color"]==last
        steps=evidence["steps"]
        assert steps and steps[0][0]==game["moves"][-1]
        stones=bits(game["moves"][:-1])
        assert replay(stones[last],stones[next_side],last,game["rule"],steps)
        replies_in_tree(stones[last],stones[next_side],last,game["rule"],steps)
        return names[last],reason
    if reason == "mutual_paired_draw":
        assert game["rule"]==1 and len(grid)>=120 and len(game["turns"])>=4
        assert all(t["eval"]=="0" for t in game["turns"][-4:])
        certificates=evidence["pairs"]
        assert [c["attacker"] for c in certificates]==[0,1]
        for c in certificates:
            attacker,pre=c["attacker"],c["preplay"]
            pairs=c["pairs"]
            endpoints=[p for pair in pairs for p in pair]
            assert all(len(pair)==2 for pair in pairs) and len(endpoints)==len(set(endpoints))
            assert pre is None or (1-attacker==next_side and pre not in endpoints)
            for p in endpoints+([] if pre is None else [pre]):
                assert 0<=p<225 and (p%15,p//15) not in grid
            for x in range(15):
                for y in range(15):
                    for dx,dy in DIRECTIONS:
                        if not cell(x+4*dx,y+4*dy): continue
                        window=[(x+i*dx,y+i*dy) for i in range(5)]
                        if any(grid.get(q)==1-attacker for q in window): continue
                        if any(grid.get((x+i*dx,y+i*dy))==attacker for i in (-1,5)): continue
                        holes={u+15*v for u,v in window if (u,v) not in grid}
                        assert pre in holes or any(a in holes and b in holes for a,b in pairs)
        return None,reason
    if reason == "mutual_draw_offer":
        assert len(game["turns"]) >= 2 and all(t["draw_offer"] for t in game["turns"][-2:])
        assert evidence["plies"] == [len(grid)-1, len(grid)]
        return None, reason
    if reason == "mutual_certified_draw":
        assert len(grid)>=120 and len(game["turns"])>=4 and all(t["eval"]=="0" for t in game["turns"][-4:])
        assert evidence["scores"] == ["0"]*4 and evidence["plies"] == list(range(len(grid)-3,len(grid)+1))
        for x in range(15):
            for y in range(15):
                for dx,dy in DIRECTIONS:
                    if cell(x+4*dx,y+4*dy):
                        segment={grid.get((x+i*dx,y+i*dy)) for i in range(5)}
                        assert 0 in segment and 1 in segment
        return None, reason
    assert reason in ("forced_five","forced_double_five")
    winner = next_side if reason=="forced_five" else 1-next_side
    def completions(color):
        wins=set()
        for p in range(225):
            xy=p%15,p//15
            if xy in grid: continue
            lengths=lines_through(grid,xy,color)
            if 5 in lengths or (game["rule"]==2 and color==1 and max(lengths)>=5):
                wins.add(p)
        return wins
    points=set(evidence["points"])
    assert evidence["color"]==winner and points<=completions(winner)
    if reason=="forced_double_five":
        assert len(points)==2 and not completions(next_side)
    else:
        assert len(points)==1
    return names[winner],reason


def verify(directory, book=None):
    fixtures=referee_fixtures()
    design=json.loads((directory/"design.json").read_bytes())
    vcpu_lanes = design.get("one_vcpu_per_game", False)
    if vcpu_lanes:
        assert design["thread_num"] == 1 and design["pondering"] is False
        slots = [lane["cpus"] for lane in design["lanes"]]
        assert all(len(cpus) == 2 and cpus[0] == cpus[1] for cpus in slots)
        assert len({cpus[0] for cpus in slots}) == len(slots)
    v2=design.get("stage")==2
    fast="opening_ids" in design
    book=book or (DATA/"v3"/design["book"] if fast and design["book"]!="openings.json" else
                  DATA/("v2/openings.json" if v2 or fast else "openings.json"))
    assert sha(book)==design["book_sha256"]
    openings = {r["id"]:r for r in json.loads(book.read_bytes())}
    if fast:
        openings={k:openings[k] for k in design["opening_ids"]}
    games = [json.loads(p.read_bytes()) for p in sorted((directory/"games").glob("*.json"))]
    assert len(games) == (2*len(openings) if fast else 600 if v2 else 200) and {(g["opening"],g["leg"]) for g in games} == {
        (o,i) for o in openings for i in (0,1)}
    elapsed, headroom, spent = [], [], {}
    diagram_moves = 0
    no_legal_black_reply = []
    for game in games:
        opening = openings[game["opening"]]
        rule = game["rule"]
        n = len(opening["moves"])
        assert game["opening_plies"] == n and game["moves"][:n] == opening["moves"] and rule == opening["rule"]
        grid = {}
        for i,p in enumerate(opening["moves"]):
            xy = p%15,p//15
            assert xy not in grid
            grid[xy] = i%2
            assert verdict(grid,xy,i%2,rule) is None
        initial,increment=design["initial_ns"],design["increment_ns"]
        assert (game["initial_ns"],game["increment_ns"])==(initial,increment)
        bank = [initial,initial]
        names = game["black"],game["white"]
        if fast:
            assert set(names) == set(design["players"])
            assert set(game["cores"].values()) == set(design["lanes"][game["lane"]]["cpus"])
        for name in names:
            expected=1<<game["cores"][name]
            for key in ("affinity_before","affinity_after"):
                assert game[key][name] and all((meta==[game["cores"][name]] if isinstance(meta,list) else
                                                   meta["effective_mask"]==expected and meta["group"]==0)
                                               for meta in game[key][name].values())
        reason,winner = None,None
        record = opening["moves"].copy()
        for i,turn in enumerate(game["turns"],n):
            assert reason is None
            side = i%2
            ns = turn["wall_ns"]
            assert ns > 0 and turn["engine"] == names[side] and turn["color"] == side
            assert bank[side] == turn["before_ns"] and bank[side]//1_000_000 == turn["info_time_left_ms"]
            elapsed.append(ns)
            if turn.get("no_coordinate") or ns > bank[side]:
                if turn.get("no_coordinate"):
                    assert ns>=bank[side] and turn["x"] is None and turn["y"] is None
                assert turn["after_ns"] == 0
                reason,winner = "time",names[1-side]
                continue
            headroom.append(bank[side]-ns)
            bank[side] += increment-ns
            assert bank[side] == turn["after_ns"]
            xy = cell(turn["x"],turn["y"])
            if xy is None or xy in grid:
                reason,winner = "illegal_coordinate",names[1-side]
                continue
            grid[xy] = side
            record.append(xy[0]+15*xy[1])
            diagram_moves += 1
            reason = verdict(grid,xy,side,rule)
            if reason:
                winner = names[1-side] if reason.startswith("forbidden") else names[side]
                if fast and reason.startswith("forbidden"):
                    del grid[xy]
                    available = False
                    for p in range(225):
                        choice = p%15, p//15
                        if choice in grid: continue
                        grid[choice] = 0
                        outcome = verdict(grid,choice,0,2)
                        del grid[choice]
                        if outcome is None or outcome == "five":
                            available = True
                            break
                    grid[xy] = 0
                    if not available:
                        no_legal_black_reply.append(dict(opening=game["opening"],leg=game["leg"],
                                                         black=names[0],ply=len(grid)))
        if reason is None and fast and game.get("adjudication"):
            assert design["adjudication"]
            winner,reason=early_verdict(grid,game,names)
        if reason is None:
            assert len(grid) == 225
            reason = "full_board"
        assert record == game["moves"] and (winner,reason) == (game["winner"],game["reason"]), game["opening"]
        timed = [0]*n + [t["wall_ns"]//1_000_000 for t in game["turns"]]
        psq = (directory/"psq"/f"{game['opening']}_{game['leg']}.psq").read_text().splitlines()
        expected_psq = ["Piskvorky 15x15, 0:0, 0"]
        expected_psq.extend(f"{p%15+1},{p//15+1},{ms}" for p,ms in zip(record,timed))
        expected_psq.extend([f"{names[0]} - {names[1]}", f"{winner or 'draw'} ({reason})"])
        assert psq == expected_psq, game["opening"]
    for opening in openings:
        pair = [g for g in games if g["opening"] == opening]
        assert pair[0]["black"] == pair[1]["white"] and pair[0]["white"] == pair[1]["black"]
        a,b=design.get("players",("orig","plus"))
        assert pair[0]["cores"][a]==pair[1]["cores"][b] and pair[0]["cores"][b]==pair[1]["cores"][a]
        assert all((g["cores"][a] == g["cores"][b]) == vcpu_lanes for g in pair)
    hardware=json.loads((directory/"hardware.json").read_bytes())
    topology={c["logical"]:c for c in hardware["cores"] if c["group"]==0 or hardware["os"].startswith("Linux")}
    highest=max(c["efficiency"] for c in topology.values())
    hybrid=len({c["efficiency"] for c in topology.values()})>1
    live,events={},[]
    for game in games:
        key=game["opening"],game["leg"]
        assert game["ended_qpc_ns"]>game["started_qpc_ns"]
        logical=list(game["cores"].values())
        if fast:
            assert not set(logical).intersection(design["exclude"])
            assert len({topology[c]["efficiency"] for c in logical})==1
            if hybrid and all("last_level_cache" in topology[c] for c in logical):
                assert len({topology[c]["last_level_cache"] for c in logical})==1
        else:
            assert all(topology[c]["efficiency"]==highest for c in logical)
        physical={(topology[c]["group"],topology[c]["physical"]) for c in logical}
        assert len(physical) == (1 if vcpu_lanes else 2)
        resources = set(logical) if vcpu_lanes else physical
        events.extend([(game["boot_time"],game["started_qpc_ns"],1,key,resources),
                       (game["boot_time"],game["ended_qpc_ns"],0,key,resources)])
    maximum=0
    for _,_,start,key,physical in sorted(events):
        if start:
            assert all(physical.isdisjoint(others) for others in live.values())
            live[key]=physical
            maximum=max(maximum,len(live))
        else:
            assert live.pop(key)==physical
    if fast:
        assert not live and maximum<=len(design["lanes"])<=5
    else:
        assert not live and maximum<=design["lanes"]<=len(hardware["performance"])//2
    result = dict(games=len(games),pairs=len(openings),timed_turns=len(elapsed),board_moves_replayed=diagram_moves,
                  hard_flags=sum(g["reason"]=="time" for g in games),min_unused_bank_ns=min(headroom, default=None),
                  longest_move_ns=max(elapsed),json_psq_mismatches=0,referee_mismatches=0,rule_fixtures=fixtures,
                  max_simultaneous_pairs=maximum)
    result["overlapping_vcpu_assignments" if vcpu_lanes else "overlapping_physical_core_assignments"] = 0
    result["illegal_moves"]=sum(g["reason"]=="illegal_coordinate" or g["reason"].startswith("forbidden") for g in games)
    if fast:
        result["no_legal_black_reply"]=no_legal_black_reply
    dump(directory/"replay.json",result)
    return games,result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-dir", type=Path, default=ROOT/"match")
    parser.add_argument("--book",type=Path,help="use the opening file supplied to the arena")
    args=parser.parse_args()
    print(verify(args.match_dir,args.book)[1])
