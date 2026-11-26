// SPDX-License-Identifier: GPL-3.0-or-later
#include "rzone.h"
#include "../../game/board.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace Search::RZone {
namespace {
constexpr int dx[] = {1, 0, 1, 1}, dy[] = {0, 1, 1, -1};
int offset(int p, int dir, int step)
{
    int x = p % Width + step * dx[dir], y = p / Width + step * dy[dir];
    return x >= 0 && x < Width && y >= 0 && y < Width ? y * Width + x : NoCell;
}
struct Segment { std::array<int, 5> cells; int before, after; };
struct Geometry {
    std::array<Segment, Segments> lines;
    std::array<std::vector<int>, Cells> through;
    Geometry()
    {
        int id = 0;
        for (int p = 0; p < Cells; ++p)
            for (int dir = 0; dir < 4; ++dir) {
                if (offset(p, dir, 4) == NoCell) continue;
                Segment &s = lines.at(id);
                for (int i = 0; i < 5; ++i) {
                    s.cells[i] = offset(p, dir, i);
                    through[s.cells[i]].push_back(id);
                }
                s.before = offset(p, dir, -1);
                s.after = offset(p, dir, 5);
                ++id;
            }
        if (id != Segments) throw std::logic_error("five-segment geometry");
    }
};
const Geometry &geometry()
{
    static const Geometry table;
    return table;
}
bool needsExact(Rule rule, Color side) { return rule == STANDARD || (rule == RENJU && side == BLACK); }

struct Placement {
    Position &pos;
    int cell;
    Placement(Position &pos, int cell, Color side) : pos(pos), cell(cell) { pos.put(cell, side); }
    ~Placement() { pos.take(cell); }
    Placement(const Placement &) = delete;
};
bool visit(Work &work) {
    if (work.nodes >= work.limit) return false;
    ++work.nodes;
    return true;
}
}  // namespace

int Mask::pop()
{
    for (int i = 0; i < 4; ++i) {
        uint64_t &w = words[i];
        if (!w) continue;
        int p = 64 * i + std::countr_zero(w);
        w &= w - 1;
        return p;
    }
    return NoCell;
}
int Mask::size() const {
    int n = 0;
    for (uint64_t w : words) n += std::popcount(w);
    return n;
}
void Mask::include(const Mask &b) { for (int i = 0; i < 4; ++i) words[i] |= b.words[i]; }
void Mask::intersect(const Mask &b) { for (int i = 0; i < 4; ++i) words[i] &= b.words[i]; }

Position::Position(Rule rule) : gameRule(rule) { cells.fill(EMPTY); }
Position::Position(const Board &board, Rule rule) : Position(rule)
{
    assert(board.size() == Width);
    for (int p = 0; p < Cells; ++p) {
        Color side = board.get(Pos(p % Width, p / Width));
        if (side != EMPTY) put(p, side);
    }
}
void Position::change(int p, Color side, int delta)
{
    for (int id : geometry().through[p]) {
        auto &n = count[id];
        uint64_t bit = uint64_t(1) << (id % 64);
        auto toggle = [&] {
            if (!n[WHITE] && (n[BLACK] == 3 || n[BLACK] == 4))
                active[BLACK][n[BLACK]-3][id/64] ^= bit;
            if (!n[BLACK] && (n[WHITE] == 3 || n[WHITE] == 4))
                active[WHITE][n[WHITE]-3][id/64] ^= bit;
        };
        toggle();
        n[side] += delta;
        toggle();
    }
}
void Position::put(int p, Color side) {
    assert(at(p) == EMPTY);
    cells[p] = side;
    change(p, side, 1);
}
void Position::take(int p) {
    Color side = cells[p];
    assert(side == BLACK || side == WHITE);
    cells[p] = EMPTY;
    change(p, side, -1);
}
Mask Position::empties() const {
    Mask empty;
    for (int p = 0; p < Cells; ++p) if (at(p) == EMPTY) empty.add(p);
    return empty;
}
Mask Position::targets(Color side, int n) const
{
    Mask result;
    const auto &index = active[side][n-3];
    for (size_t i = 0; i < index.size(); ++i)
        for (uint64_t bits = index[i]; bits; bits &= bits-1) {
            const Segment &s = geometry().lines[64*i + std::countr_zero(bits)];
            if (needsExact(gameRule, side) && (at(s.before) == side || at(s.after) == side))
                continue;
            for (int p : s.cells) if (at(p) == EMPTY) result.add(p);
        }
    return result;
}
Color Position::read(int p, Mask *trace) const {
    if (trace && p != NoCell) trace->add(p);
    return at(p);
}
int Position::run(int p, Color side, int dir, Mask *trace) const {
    int n = 1;
    for (int sign : {-1, 1})
        for (int i = sign; read(offset(p, dir, i), trace) == side; i += sign) ++n;
    return n;
}
bool Position::winsOn(int p, Color side) const {
    for (int d = 0; d < 4; ++d) {
        int n = run(p, side, d, nullptr);
        if (needsExact(gameRule, side) ? n == 5 : n >= 5) return true;
    }
    return false;
}

Position::BlackStatus Position::blackStatus(int p, Mask *trace)
{
    bool overline = false;
    for (int d = 0; d < 4; ++d) {
        int n = run(p, BLACK, d, trace);
        if (n == 5) return Five;
        if (n > 5) overline = true;
    }
    if (overline) return Foul;

    // Identify fours by their four stones. The two wins of one straight four
    // are one four. Two different fours can be on the same line.
    Mask firstFour;
    for (int id : geometry().through[p]) {
        const Segment &s = geometry().lines[id];
        Mask stones;
        int black = 0;
        bool blocked = false;
        for (int q : s.cells) {
            Color c = read(q, trace);
            if (c == WHITE) { blocked = true; break; }
            if (c == BLACK) { stones.add(q); ++black; }
        }
        if (blocked || black != 4 || read(s.before, trace) == BLACK || read(s.after, trace) == BLACK)
            continue;
        if (firstFour.any() && firstFour != stones) return Foul;
        firstFour = stones;
    }

    // A real three admits a legal non-winning extension to a straight four.
    // Only windows containing the newly placed stone can make a new three.
    struct Three { Mask stones; int extension; };
    std::array<Three, 16> possible;
    int npossible = 0;
    for (int d = 0; d < 4; ++d)
        for (int start = -3; start <= 0; ++start) {
            if (read(offset(p,d,start-1),trace) != EMPTY || read(offset(p,d,start+4),trace) != EMPTY)
                continue;
            Mask stones;
            int extension = NoCell, black = 0;
            for (int i = start; i < start+4; ++i) {
                int q = offset(p,d,i);
                Color c = read(q,trace);
                if (c == BLACK) { stones.add(q); ++black; }
                else if (c == EMPTY) extension = q;
                else break;
            }
            if (black == 3 && extension != NoCell && read(offset(p,d,start-2),trace) != BLACK
                && read(offset(p,d,start+5),trace) != BLACK)
                possible[npossible++] = {stones, extension};
        }
    if (npossible < 2) return Allowed;
    bool distinct = false;
    for (int i = 1; i < npossible; ++i) distinct |= possible[0].stones != possible[i].stones;
    if (!distinct) return Allowed;
    Mask firstReal;
    for (int i = 0; i < npossible; ++i) {
        const Three &t = possible[i];
        if (firstReal == t.stones) continue;
        Placement extend(*this, t.extension, BLACK);
        if (blackStatus(t.extension, trace) != Allowed) continue;
        if (firstReal.any()) return Foul;
        firstReal = t.stones;
    }
    return Allowed;
}
bool Position::forbidden(int p, Color side, Mask *trace) {
    if (gameRule != RENJU || side != BLACK) return false;
    Placement tryMove(*this, p, BLACK);
    return blackStatus(p, trace) == Foul;
}

namespace {
Proof afterAttack(Position &pos, Color attacker, int move, int plies, Work &work)
{
    if (plies < 3 || !visit(work) || pos.forbidden(move, attacker)) return {};
    Placement attack(pos, move, attacker);
    Color defender = ~attacker;
    if (pos.targets(defender, 4).any()) return {};
    Mask wins = pos.targets(attacker, 4);
    int block = wins.pop();
    if (block == NoCell) return {};
    int other = wins.pop();
    if (other != NoCell) return {{move, block, other, End::Fork}};
    if (pos.forbidden(block, defender)) return {{move, block, NoCell, End::ForbiddenBlock}};
    Placement parry(pos, block, defender);
    Proof tail = findWin(pos, attacker, plies-2, work);
    if (!tail.empty()) tail.insert(tail.begin(), {move, block, NoCell, End::Forced});
    return tail;
}

bool replay(Position &pos, Color a, const Proof &proof, size_t idx, Mask *trace)
{
    const Step &s = proof[idx];
    if (s.attack < 0 || s.attack >= Cells || pos.at(s.attack) != EMPTY) return false;
    if (trace) trace->add(s.attack);
    bool last = idx+1 == proof.size();
    if (s.kind == End::Five) {
        Placement m(pos, s.attack, a);
        return last && pos.winsOn(s.attack, a);
    }
    if (pos.forbidden(s.attack, a, trace)) return false;
    Placement m(pos, s.attack, a);
    Color d = ~a;
    if (pos.winsOn(s.attack, a) || pos.targets(d, 4).any()) return false;
    if (s.block < 0 || s.block >= Cells || pos.at(s.block) != EMPTY) return false;
    {
        Placement completion(pos, s.block, a);
        if (!pos.winsOn(s.block, a)) return false;
    }
    if (trace) {
        trace->add(s.block);
        // With one added defender stone, a new immediate counterwin must use
        // that stone and one other empty point of a three-stone segment.
        trace->include(pos.targets(d, 3));
    }
    if (s.kind == End::Fork) {
        if (!last || s.other < 0 || s.other >= Cells || s.other == s.block || pos.at(s.other) != EMPTY)
            return false;
        if (trace) trace->add(s.other);
        Placement second(pos, s.other, a);
        return pos.winsOn(s.other, a);
    }
    bool foul = pos.forbidden(s.block, d, trace);
    if (s.kind == End::ForbiddenBlock) return last && foul;
    if (last || foul || s.kind != End::Forced) return false;
    Placement reply(pos, s.block, d);
    return replay(pos, a, proof, idx+1, trace);
}
}  // namespace

Proof findWin(Position &pos, Color a, int plies, Work &work)
{
    if (plies < 1 || !visit(work)) return {};
    int win = pos.targets(a,4).pop();
    if (win != NoCell) return {{win, NoCell, NoCell, End::Five}};
    if (plies < 3) return {};
    Mask attacks = pos.targets(a,3);
    for (int m; work.nodes < work.limit && (m=attacks.pop()) != NoCell;) {
        Proof proof = afterAttack(pos,a,m,plies,work);
        if (!proof.empty()) return proof;
    }
    return {};
}
bool verify(Position &pos, Color attacker, const Proof &proof, Mask *zone)
{
    if (proof.empty()) return false;
    Mask trace;
    if (zone) trace = pos.targets(~attacker, 4);  // D may win *before* A starts.
    if (!replay(pos,attacker,proof,0,zone ? &trace : nullptr)) return false;
    if (zone) {
        trace.intersect(pos.empties());
        *zone = trace;
    }
    return true;
}
void Mustplay::add(const Certificate &cert) {
    if (valid) intersection.intersect(cert.zone);
    else intersection = cert.zone;
    valid = true;
}
DefenseMap defend(Position &parent, Color defender, int plies, bool useZones, bool audit)
{
    const Color attacker = ~defender;
    DefenseMap result;
    Work work;
    Mustplay must;
    std::vector<Certificate> promoted;
    Mask empty = parent.empties();
    for (int move; (move = empty.pop()) != NoCell;) {
        if (parent.forbidden(move, defender))
            continue;
        result.legal.add(move);
        if (must.outside(move)) {
            result.losses.add(move);
            ++result.skipped;
            if (audit) {
                Placement reply(parent, move, defender);
                bool witnessed = false;
                for (const Certificate &cert : promoted) {
                    if (cert.zone.has(move))
                        continue;
                    if (parent.winsOn(move, defender) || !verify(parent, attacker, cert.proof))
                        throw std::logic_error("invalid outside-zone replay");
                    witnessed = true;
                    ++result.replays;
                }
                if (!witnessed)
                    throw std::logic_error("missing certificate");
            }
            continue;
        }
        Proof proof;
        {
            ++work.nodes;
            Placement reply(parent, move, defender);
            if (!parent.winsOn(move, defender))
                proof = findWin(parent, attacker, plies, work);
        }
        if (proof.empty())
            continue;
        result.losses.add(move);
        // First solve an actual child. Promote its proof only if it also
        // replays on the common parent and another sibling still exists.
        if (!useZones || !empty.any() || promoted.size() >= 4)
            continue;
        if (std::any_of(promoted.begin(), promoted.end(), [&](const Certificate &cert) {
                return cert.proof.front().attack == proof.front().attack;
            }))
            continue;
        Mask zone;
        if (!verify(parent, attacker, proof, &zone))
            continue;
        promoted.push_back({std::move(proof), zone});
        must.add(promoted.back());
        ++result.promoted;
    }
    result.nodes = work.nodes;
    return result;
}
}  // namespace Search::RZone
