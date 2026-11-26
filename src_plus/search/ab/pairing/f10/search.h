// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "cheap.h"
#include "../lazy.h"

namespace f10 {

constexpr int FirstSearchPly = 90;
constexpr int RootLiveCap = 112;

// Root construction is amortized over the entire search. Ordinary disjoint
// pairs suffice for both early fixtures. Forced short pairs are removed before
// the remaining edges compete for two distinct representatives each.
inline bool rootDefence(const Board &board, int attacker, bool first, Proof &proof)
{
    if (board.finished || board.ply < FirstSearchPly || board.ply >= Cells
        || board.count(attacker) > RootLiveCap)
        return false;
    const bool blackGuard = board.rule == f8::Renju && attacker == 1;
    if (blackGuard) {
        auto cheap = cheapBlack(board, first);
        if (cheap.found) {
            proof = std::move(cheap.proof);
            return true;
        }
    }
    f8::Family family(board, attacker);
    auto edges = minimal(Edges(family.edges.begin(), family.edges.begin() + family.size()));
    Preferences pref(edges);
    Diagnosis d;
    d.pre = preplays(edges, first, board.empty, pref);
    if (d.pre.empty()) return false;
    for (int pre : d.pre) {
        const int hit = pre < 0 ? 0 : pref.degree[pre];
        if ((pre >= 0 && !hit) || 2 * (int(edges.size()) - hit) > pref.twicePairCapacity)
            continue;
        if (pairCover(edges, pre, pref, d.proof.pairs)
            || pairCover(edges, pre, pref, d.proof.pairs, 1)
            || reducedPairing(residual(edges, pre), d.proof.pairs)) {
            d.proof.pre = pre;
            d.ordinary = true;
            break;
        }
    }
    if (!d.ordinary) return false;
    if (blackGuard) {
        d.t = transversal(edges);
        defendBlack(board, edges, first, d);
        if (!d.accepted) return false;
    }
    if (!verify(board, attacker, first, d.proof)) return false;
    proof = std::move(d.proof);
    return true;
}

inline bool rootDraw(const Board &board, Proof &firstDefence)
{
    if (board.rule == f8::Free || board.finished || board.ply < FirstSearchPly
        || board.ply >= Cells || std::max(board.count(0), board.count(1)) > RootLiveCap)
        return false;
    Proof secondDefence;
    const int first = board.ply % 2;
    return rootDefence(board, first, false, secondDefence)
           && rootDefence(board, 1-first, true, firstDefence);
}

inline f8::Bounds fastBounds(const Board &board, bool lower, bool upper)
{
    if (board.rule == f8::Free || board.finished || board.ply < FirstSearchPly
        || board.ply >= Cells)
        return {};
    // Keep the more expensive rank-two inference at its v3 threshold.
    auto bounds = board.ply >= f8::MinPly ? f8::lazy(board, lower, upper)
                                          : f8::prove(board, lower, upper, f8::MaxLive, false);
    if (board.rule == f8::Renju && !bounds.blackFree) {
        const bool first = board.ply % 2 == 0;
        if (((first && lower && bounds.lo < 0) || (!first && upper && bounds.hi > 0))
            && cheapBlack(board, first).found) {
            if (first) bounds.lo = 0;
            else bounds.hi = 0;
        }
    }
    return bounds;
}
} // namespace f10
