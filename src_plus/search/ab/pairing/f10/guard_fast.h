// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "certificates.h"
#include <bitset>
namespace f10 {
struct Board : f8::Position {
    f8::Live whiteFree;
    explicit Board(f8::Rule rule=f8::Standard): f8::Position(rule), whiteFree(live[0]) {}
    void put(int p,int color) {
        if (rule==f8::Renju && color==1)
            for (int i=0;i<9;++i) whiteFree[i]&=~f8::geometry.interior[p][i];
        f8::Position::put(p,color);
    }
};
using Mask=std::bitset<Cells>;
struct Span {
    Mask five, core;
    int beyond=NoMove;
};
inline const auto spans=[]{
    std::array<Span,f8::Windows> index{};
    for (int i=0;i<f8::Windows;++i) {
        auto& e=f8::geometry.lines[i];auto& s=index[i];
        for (int p:e.p) s.five.set(p);
        s.core=s.five;s.core.reset(e.p[0]);
        if (e.flank[1]!=f8::None) {
            int last=e.flank[1],before=e.p[4];
            int x=2*(last%15)-before%15, y=2*(last/15)-before/15;
            if (f8::onBoard(x,y)) s.beyond=15*y+x;
        }
    }
    return index;
}();
// One open five can host at most one four at an instant. A married pair in
// the five prevents all fours there: it is either still empty on both ends
// or already contains a white stone. The corresponding test on the interior
// of an open six prevents every three extendible to a straight four there.
inline bool quickGuard(const Board& b,const Proof& proof,const std::array<int,Cells>& mate) {
    Mask seenFour, seenThree, roots;
    for (int i=0;i<Cells;++i) if (b.empty.has(i)) roots.set(i);
    auto black=[&](int p){return p==proof.pre || (p!=f8::None && p>=0 && !b.empty.has(p) && !b.white.has(p));};
    for (int w=0;w<9;++w) for (auto live=b.whiteFree[w];live;live&=live-1) {
        int id=w*64+std::countr_zero(live);
        const auto& e=f8::geometry.lines[id];const auto& s=spans[id];
        int last=e.flank[1];bool pairFive=false,pairCore=false,pairSix=false;
        for (int p:e.p) if (int q=mate[p];q>=0) {
            if (s.five[q]) {pairFive=true;pairCore|=p!=e.p[0] && q!=e.p[0];}
            if (q==last) pairSix=true;
        }
        if (!pairFive && !black(e.flank[0]) && !black(last)) {
            auto overlap=s.five & roots;
            if ((overlap & seenFour).any()) return false;
            seenFour|=overlap;
        }
        if (last==f8::None || b.white.has(last)) continue;
        if (!pairFive && !pairSix) return false;
        if (pairCore || black(e.p[0]) || black(last) || black(e.flank[0]) ||
            (s.beyond>=0 && black(s.beyond))) continue;
        auto overlap=s.core & roots;
        if ((overlap & seenThree).any()) return false;
        seenThree|=overlap;
    }
    return true;
}
} // namespace f10
