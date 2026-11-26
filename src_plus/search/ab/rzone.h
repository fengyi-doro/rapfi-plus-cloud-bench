// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "../../core/pos.h"
#include "../../core/types.h"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

class Board;

namespace Search::RZone {

constexpr int Width = 15, Cells = Width * Width, Segments = 572;
constexpr int NoCell = -1;
inline int cellId(Pos p) { return p.y() * Width + p.x(); }

struct Mask {
    std::array<uint64_t, 4> words {};
    void add(int p) { words[p / 64] |= uint64_t(1) << (p % 64); }
    bool has(int p) const { return (words[p / 64] >> (p % 64)) & 1; }
    int pop();  // Remove and return the first cell or NoCell if empty.
    bool any() const { return words[0] || words[1] || words[2] || words[3]; }
    int size() const;
    void include(const Mask &other);
    void intersect(const Mask &other);
    bool operator==(const Mask &) const = default;
};

// Geometry separate from Rapfi's move picker, pattern LUT, evaluator and TT.
class Position {
public:
    explicit Position(Rule rule);
    Position(const Board &board, Rule rule);
    Color at(int p) const { return p == NoCell ? WALL : cells[p]; }
    Rule rule() const { return gameRule; }
    void put(int p, Color side);
    void take(int p);
    Mask empties() const;
    // Empty points of unblocked five-cell segments with exactly n stones (3 or 4).
    Mask targets(Color side, int n) const;
    bool winsOn(int p, Color side) const;  // After placing at p.
    bool forbidden(int p, Color side, Mask *reads = nullptr);  // Before placing.

private:
    Rule gameRule;
    std::array<Color, Cells> cells;
    std::array<std::array<uint8_t, 2>, Segments> count {};
    using SegmentBits = std::array<uint64_t, (Segments + 63) / 64>;
    SegmentBits active[2][2] {};  // side and stone count minus three
    void change(int p, Color side, int delta);
    Color read(int p, Mask *trace) const;
    int run(int p, Color side, int dir, Mask *trace) const;
    enum BlackStatus { Allowed, Five, Foul };
    BlackStatus blackStatus(int p, Mask *trace);  // Black stone already on p.
};

enum class End { Five, Forced, Fork, ForbiddenBlock };
struct Step { int attack, block, other; End kind; };
using Proof = std::vector<Step>;
struct Certificate { Proof proof; Mask zone; };

// Stand-alone tactical query. An exhausted budget means 'no certificate returned'.
// An unlimited budget is used for the complete bounded-VCF reference predicate.
struct Work { uint64_t nodes = 0, limit = UINT64_MAX; };
Proof findWin(Position &pos, Color attacker, int plies, Work &work);
bool verify(Position &pos, Color attacker, const Proof &proof, Mask *zone = nullptr);

struct Mustplay {
    Mask intersection;
    bool valid = false;
    void add(const Certificate &cert);
    bool outside(int cell) const { return valid && !intersection.has(cell); }
};

struct DefenseMap {
    Mask legal, losses;
    uint64_t nodes = 0;
    unsigned skipped = 0, replays = 0, promoted = 0;
};
// Complete bounded VCF defense query. Certificates belong to this parent only.
DefenseMap defend(Position &parent, Color defender, int plies, bool useZones, bool audit = false);

}  // namespace Search::RZone
