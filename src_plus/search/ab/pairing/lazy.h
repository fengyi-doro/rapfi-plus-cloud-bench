// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "prover.h"
namespace f8 {
constexpr int MinPly=120, MaxLive=64, RankCap=24;
inline Bounds lazy(const Position &b, bool lower, bool upper, bool rankFallback=true) {
    if (b.finished || b.ply<MinPly) return {};
    auto r=prove(b,lower,upper,MaxLive,false);
    if (!rankFallback || ((!lower || r.pairLo) && (!upper || r.pairHi))) return r;
    if (b.count(0)>RankCap || b.count(1)>RankCap) return r;
    int self=b.ply%2, opp=1-self;
    bool unrestricted=b.rule!=Renju || r.blackFree;
    lower=lower && !r.pairLo && (self || unrestricted);
    upper=upper && !r.pairHi && (opp || unrestricted);
    if (!lower && !upper) return r;
    Family mine(b,self),other(b,opp);
    if (lower) r.lo=outcome(b.empty,stableGraph(mine),easierGraph(other));
    if (upper) r.hi=outcome(b.empty,easierGraph(mine),stableGraph(other));
    if (r.lo>r.hi) throw std::logic_error("inconsistent lazy bounds");
    return r;
}
} // namespace f8
