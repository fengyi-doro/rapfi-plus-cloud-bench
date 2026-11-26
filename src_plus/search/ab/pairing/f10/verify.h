// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "renju.h"
#include "guard_fast.h"
namespace f10 {

struct Form { Mask black, mandatoryWhite, holes, flank; };
inline bool guardHolds(const Board& board,const Proof& proof,const std::array<int,Cells>& mate) {
    Mask fixedBlack, fixedWhite, initiallyEmpty;
    for (int p=0;p<Cells;++p) {
        if (board.empty.has(p)) initiallyEmpty.set(p);
        else (board.white.has(p)?fixedWhite:fixedBlack).set(p);
    }
    if (proof.pre>=0) fixedBlack.set(proof.pre);
    std::array<std::array<std::vector<Form>,Cells>,2> at;
    auto insert=[&](Mask stones,Mask forced,Mask holes,Mask flanks,const auto& line,int kind) {
        if ((forced & (stones|holes)).any() || (fixedBlack & (holes|flanks)).any() ||
            (fixedWhite & (stones|holes)).any()) return true;
        Form form{stones,forced,holes,flanks};
        for (int p:line) if (initiallyEmpty[p] && stones[p]) {
            for (auto& prev:at[kind][p]) {
                if (prev.black==stones) continue; // alternate completions of the same four/three
                if (((prev.mandatoryWhite|forced) & (prev.black|stones|prev.holes|holes)).any()) continue;
                if (((prev.black|stones) & (flanks|prev.flank|holes|prev.holes)).any()) continue;
                return false;
            }
            at[kind][p].push_back(form);
        }
        return true;
    };
    for (int i=0;i<9;++i) for (auto live=board.whiteFree[i];live;live&=live-1) {
        const auto& line=f8::geometry.lines[64*i+std::countr_zero(live)];
        Mask five, flank, pairsWhite;
        for (int p:line.p) {five.set(p); if (mate[p]>=0) pairsWhite.set(mate[p]);}
        if ((five & fixedWhite).any()) continue;
        for (int p:line.flank) if (p!=f8::None) flank.set(p);
        if (!(flank & fixedBlack).any()) for (int h:line.p) if (board.empty.has(h) && h!=proof.pre) {
            Mask support=five, forced=pairsWhite, hole;
            support.reset(h); hole.set(h); if (mate[h]>=0) forced.reset(mate[h]);
            if (!insert(support,forced,hole,flank,line.p,0)) return false;
        }
        int last=line.flank[1];
        if (last==f8::None || fixedWhite[last]) continue;
        Mask six=five, forcedSix=pairsWhite; six.set(last); if (mate[last]>=0) forcedSix.set(mate[last]);
        if (!(six & forcedSix).any()) return false;
        int start=line.p[0];
        if (fixedBlack[start] || fixedBlack[last]) continue;
        Mask outside;
        if (line.flank[0]!=f8::None) outside.set(line.flank[0]);
        int x=last%15+(last%15-line.p[4]%15), y=last/15+(last/15-line.p[4]/15);
        if (f8::onBoard(x,y)) outside.set(15*y+x);
        if ((outside & fixedBlack).any()) continue;
        for (int j=1;j<=4;++j) {
            int h=line.p[j]; if (fixedBlack[h]) continue;
            Mask holes; holes.set(start); holes.set(last); holes.set(h);
            Mask stones=six & ~holes, forced=forcedSix;
            for (int k:{start,last,h}) if (mate[k]>=0) forced.reset(mate[k]);
            if (!insert(stones,forced,holes,outside,line.p,1)) return false;
        }
    }
    return true;
}
inline bool verify(const Board& b,int attacker,bool first,const Proof& proof,bool* quick=nullptr) {
    std::array<int,Cells> mate; mate.fill(NoMove);
    std::array<bool,Cells> reserved{};
    auto reserve=[&](int p){
        if (unsigned(p)>=Cells || reserved[p] || !b.empty.has(p)) return false;
        reserved[p]=true;return true;
    };
    auto live=b.live[attacker];
    auto remove=[&](const auto& sites) {
        for (int i=0;i<9;++i) {
            uint64_t covered=~uint64_t(0);
            for (int p:sites) covered &= f8::geometry.interior[p][i];
            live[i] &= ~covered;
        }
    };
    if (proof.pre>=0) {
        if (!first || !reserve(proof.pre)) return false;
        remove(std::array{proof.pre});
    }
    for (auto [a,c]:proof.pairs) {
        if (!reserve(a) || !reserve(c)) return false;
        mate[a]=c; mate[c]=a; remove(std::array{a,c});
    }
    for (auto& block:proof.blocks) {
        Points domain=active(block.protect);
        for (int p:domain) if (!reserve(p)) return false;
        for (auto& e:block.protect) {if (!e.n) return false; remove(points(e));}
        if (block.potential) {if (!potential(block.protect)) return false; continue;}
        if (block.branches.size()!=domain.size()) return false;
        std::array<bool,Cells> handled{};
        for (auto& branch:block.branches) {
            int m=branch.attack, z=branch.reply;
            if (m==z || !contains(domain,m) || !contains(domain,z) || handled[m]) return false;
            handled[m]=true;
            std::array<int,Cells> local; local.fill(NoMove);
            for (auto [a,c]:branch.pairs) {
                if (a==c || a==m || c==m || a==z || c==z || !contains(domain,a) || !contains(domain,c) ||
                    local[a]>=0 || local[c]>=0) return false;
                local[a]=c; local[c]=a;
            }
            for (auto& e:block.protect) {
                bool hit=has(e,z);
                for (int p:points(e)) if (local[p]>=0 && has(e,local[p])) hit=true;
                if (!hit || (e.n==1 && has(e,m))) return false;
            }
        }
    }
    if (std::any_of(live.begin(),live.end(),[](auto w){return w!=0;})) return false;
    if (b.rule!=f8::Renju || attacker!=1) return !proof.mirror;
    if (!proof.mirror) return b.allFutureBlackMovesLegal();
    if (!proof.blocks.empty() || first!=(proof.pre>=0)) return false;
    for (int p=0;p<Cells;++p) if (b.empty.has(p) && p!=proof.pre && mate[p]<0) return false;
    if (quickGuard(b,proof,mate)) { if (quick) *quick=true; return true; }
    return guardHolds(b,proof,mate);
}
} // namespace f10
