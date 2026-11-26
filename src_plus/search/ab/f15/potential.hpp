// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "renju_vcf.hpp"
#include <utility>
namespace f15 {
using Edges=std::vector<Mask>;
using Pair=std::pair<int,int>;
using Pairs=std::vector<Pair>;

inline Edges liveEdges(const Board& b,int rule,int maker) {
    Edges result;
    auto own=b.stones(maker),enemy=b.stones(1-maker);
    bool exact=rule==1 || (rule==2 && maker==0);
    for(const auto& w:geometry.fives) {
        if((w.body&enemy).any() || (exact && (w.ends&own).any())) continue;
        result.push_back(w.body&~own);
    }
    return result;
}
// A strict strategy that blocks every minimal member blocks every member.
inline Edges minimal(Edges rows) {
    std::stable_sort(rows.begin(),rows.end(),[](auto a,auto b){return a.count()<b.count();});
    Edges result;std::array<std::vector<int>,Cells> byFirst;
    for(auto row:rows) {
        if(!row.any()) return {row};
        auto points=row;bool covered=false;int first=points.pop(),p=first;
        do {
            for(int i:byFirst[p]) if(row.contains(result[i])) {covered=true;break;}
            p=points.pop();
        } while(!covered && p>=0);
        if(!covered) {byFirst[first].push_back(int(result.size()));result.push_back(row);}
    }
    return result;
}
struct Potential {
    int sum32=0,load32=0;
    std::array<int,Cells> load{};
    explicit Potential(const Edges& rows,int breaker=-1) {
        for(auto e:rows) {
            if(breaker>=0 && e.has(breaker)) continue;
            int w=32>>e.count();
            sum32+=w;
            while(e.any()) {int q=e.pop();load32=std::max(load32,load[q]+=w);}
        }
    }
    bool timed(bool makerFirst) const {return sum32+(makerFirst?load32:0)<32;}
    // Explicit first Breaker placement, then verify the Maker-first condition.
    bool withFirstMove(const Edges& rows,bool makerFirst,int& move) const {
        move=-1;
        if(timed(makerFirst)) return true;
        if(makerFirst || sum32-load32>=32) return false;
        for(int p=0;p<Cells;++p) if(sum32-load[p]<32 && Potential(rows,p).timed(true)) {
            move=p;return true;
        }
        return false;
    }
};
// Sufficient future legality for every Black reply and tempo. White barriers
// only accumulate. A true three needs a segment of at least six squares.
inline bool allFutureBlackLegal(const Board& b) {
    auto free=geometry.playable&~b.occupied();
    while(free.any()) {
        int p=free.pop(),possibleFourDirections=0;
        for(int d=0;d<4;++d) {
            int length=1;
            for(int sign:{-1,1}) for(int t=sign;;t+=sign) {
                int q=offset(p,d,t);
                if(q<0 || b.white.has(q)) break;
                if(++length>=6) return false;
            }
            if(length==5 && ++possibleFourDirections>=2) return false;
        }
    }
    return true;
}
inline bool legalStrategy(const Board& b,int rule,int maker) {
    return rule!=2 || maker==0 || allFutureBlackLegal(b);
}
// Each sign bounds the actual strong game. Call only after checking a root is
// nonterminal. sideToMove, maker, and the meaning of the sign are independent.
inline bool esBound(const Board& b,int rule,int maker,int sideToMove) {
    auto rows=minimal(liveEdges(b,rule,maker));int pre;
    return legalStrategy(b,rule,maker) && Potential(rows).withFirstMove(rows,sideToMove==maker,pre);
}
inline bool verifyMixed(const Board& b,int rule,int maker,const Edges& family,const Pairs& pairs) {
    // family must be the exact root family from liveEdges or its minimal antichain.
    Mask used;
    for(auto [a,c]:pairs) {
        if(unsigned(a)>=Cells || unsigned(c)>=Cells || a==c || used.has(a) || used.has(c)
            || b.occupied().has(a) || b.occupied().has(c)) return false;
        used.add(a);used.add(c);
    }
    if(!legalStrategy(b,rule,maker)) return false;
    Edges projected;
    for(auto row:family) {
        if(std::any_of(pairs.begin(),pairs.end(),[&](Pair pair){return row.has(pair.first)&&row.has(pair.second);})) continue;
        projected.push_back(row&~used);
    }
    return Potential(minimal(std::move(projected))).timed(true);
}
// Host-facing entry accepts no caller-supplied edge list.
inline bool mixedBound(const Board& b,int rule,int maker,const Pairs& proposal) {
    return verifyMixed(b,rule,maker,liveEdges(b,rule,maker),proposal);
}
} // namespace f15
