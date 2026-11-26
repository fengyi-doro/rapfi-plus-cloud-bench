// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "verify.h"
namespace f10 {
struct CheapResult {
    bool tried=false, found=false;
    Proof proof;
};
struct Corridor {
    std::array<uint8_t,6> cell{};
    int length=0, freeCount=0;
};
// Fill the unassigned points of an ordinary white-window strategy. A pair
// inside a six blocks an overline, and an interior pair also seals its threes.
inline bool completeCorridors(const Board& b,Proof& p) {
    std::array<int,Cells> mate; mate.fill(NoMove);
    if (p.pre>=0) mate[p.pre]=p.pre;
    for (auto [a,z]:p.pairs) {mate[a]=z;mate[z]=a;}
    auto available=[&](int v){return b.empty.has(v) && mate[v]<0;};
    auto bind=[&](int a,int z){mate[a]=z;mate[z]=a;p.pairs.emplace_back(a,z);};
    std::vector<Corridor> corridors;
    for (int w=0;w<9;++w) for (auto bits=b.whiteFree[w];bits;bits&=bits-1) {
        auto& line=f8::geometry.lines[64*w+std::countr_zero(bits)];
        Corridor c;c.length=5;
        std::copy(line.p.begin(),line.p.end(),c.cell.begin());
        for (int i=0;i<5;++i) c.freeCount+=available(c.cell[i]);
        corridors.push_back(c);
        int last=line.flank[1];
        if (last!=f8::None && !b.white.has(last)) {
            c.length=6;c.cell[5]=last;c.freeCount+=available(last);corridors.push_back(c);
        }
    }
    std::stable_sort(corridors.begin(),corridors.end(),[](auto& a,auto& z){
        return a.length!=z.length?a.length>z.length:a.freeCount<z.freeCount;
    });
    for (auto& c:corridors) {
        bool sealed=false;int a=-1,z=-1,best=-100;
        for (int i=0;i<c.length && !sealed;++i) for (int j=0;j<i;++j) {
            int u=c.cell[i],v=c.cell[j];
            if (mate[u]==v) {sealed=true;break;}
            if (!available(u) || !available(v)) continue;
            int core=(i<c.length-1 && j>0);
            int score=8*core-(i-j);
            if (score>best) {best=score;a=u;z=v;}
        }
        if (sealed) continue;
        if (a>=0) bind(a,z);
        else if (c.length==6) return false;
    }
    int spare=-1;
    for (int v=0;v<Cells;++v) if (available(v)) {
        if (spare<0) spare=v;
        else {bind(spare,v);spare=-1;}
    }
    return spare<0;
}
// Independent entry point from an incremental board. No normalized family,
// archived label or heavyweight witness is needed. Always verifies its result.
inline CheapResult cheapBlack(const Board& b,bool first) {
    CheapResult r;
    if (b.count(1)>64 || b.allFutureBlackMovesLegal()) return r;
    r.tried=true;
    f8::Family white(b,1);
    Proof& p=r.proof;p.mirror=true;
    if (first && white.size()>0) {
        for (int v:points(white.edges[0])) {
            bool hits=true;
            for (int i=1;i<white.size() && hits;++i) hits=has(white.edges[i],v);
            if (hits) {p.pre=v;break;}
        }
    }
    if (p.pre<0) {
        f8::Witness w;
        if (!f8::pairing(white,first,w)) return r;
        if (w.preplay!=f8::None) p.pre=w.preplay;
        for (int v=0;v<Cells;++v) if (w.mate[v]!=f8::None && v<w.mate[v])
            p.pairs.emplace_back(v,w.mate[v]);
        if (first && p.pre<0) {
            for (int v=0;v<Cells;++v) if (b.empty.has(v) && w.mate[v]==f8::None) {p.pre=v;break;}
            if (p.pre<0) return r;
        }
    }
    if (completeCorridors(b,p)) r.found=verify(b,1,first,p);
    return r;
}
} // namespace f10
