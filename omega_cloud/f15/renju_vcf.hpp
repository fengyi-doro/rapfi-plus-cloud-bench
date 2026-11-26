// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace f15 {
constexpr int Width = 15, Cells = 225;
struct Mask {
    std::array<uint64_t,4> w{};
    bool operator==(const Mask&) const = default;
    static int bit(int p) { return p + p/Width; }
    bool has(int p) const { int i=bit(p); return (w[i/64]>>(i%64))&1; }
    void add(int p) { int i=bit(p); w[i/64] |= uint64_t(1)<<(i%64); }
    void remove(int p) { int i=bit(p); w[i/64] &= ~(uint64_t(1)<<(i%64)); }
    bool any() const { return w[0] || w[1] || w[2] || w[3]; }
    int count() const { int n=0; for (auto v:w) n+=std::popcount(v); return n; }
    int pop() {
        for (int i=0;i<4;++i) if (w[i]) {
            int b=64*i+std::countr_zero(w[i]);
            w[i] &= w[i]-1;
            return b-b/16;
        }
        return -1;
    }
    Mask operator&(const Mask& m) const { Mask s; for(int i=0;i<4;++i) s.w[i]=w[i]&m.w[i]; return s; }
    Mask operator|(const Mask& m) const { Mask s; for(int i=0;i<4;++i) s.w[i]=w[i]|m.w[i]; return s; }
    Mask operator~() const { Mask s; for(int i=0;i<4;++i) s.w[i]=~w[i]; return s; }
    Mask right(int n) const {
        Mask s; int whole=n/64, sub=n%64;
        for(int i=0;i+whole<4;++i) {
            s.w[i]=w[i+whole]>>sub;
            if(sub && i+whole+1<4) s.w[i]|=w[i+whole+1]<<(64-sub);
        }
        return s;
    }
    Mask left(int n) const {
        Mask s; int whole=n/64, sub=n%64;
        for(int i=whole;i<4;++i) {
            s.w[i]=w[i-whole]<<sub;
            if(sub && i>whole) s.w[i]|=w[i-whole-1]>>(64-sub);
        }
        return s;
    }
    bool contains(const Mask& m) const { return (*this&m)==m; }
};
struct Board {
    Mask black,white;
    bool operator==(const Board&) const = default;
    Mask occupied() const {return black|white;}
    Mask stones(int side) const {return side?white:black;}
    Board with(int p,int side) const {Board b=*this; (side?b.white:b.black).add(p);return b;}
};
struct BoardHash {
    size_t operator()(const Board& b) const {
        uint64_t h=0x9e3779b97f4a7c15ULL;
        for(const auto* m:{&b.black,&b.white}) for(auto x:m->w)
            h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        return size_t(h);
    }
};
inline constexpr int directions[4][2]={{1,0},{0,1},{1,1},{-1,1}};
inline int offset(int p,int d,int t) {
    int x=p%Width+t*directions[d][0],y=p/Width+t*directions[d][1];
    return unsigned(x)<Width && unsigned(y)<Width? y*Width+x : -1;
}
struct Segment { Mask body,ends,beyond; };
struct Geometry {
    Mask playable;
    std::array<Mask,4> starts;
    std::array<Mask,Cells> rays;
    std::vector<Segment> fives;
    std::array<std::array<std::vector<Segment>,3>,Cells> through;
    Geometry() {
        for(int p=0;p<Cells;++p) {
            playable.add(p);
            for(int d=0;d<4;++d) {
                for(int t=-5;t<=5;++t) if(int q=offset(p,d,t);q>=0) rays[p].add(q);
                for(int n:{4,5,6}) {
                    if(offset(p,d,n-1)<0) continue;
                    Segment s;
                    for(int i=0;i<n;++i) s.body.add(offset(p,d,i));
                    for(int t:{-1,n}) if(int q=offset(p,d,t);q>=0) s.ends.add(q);
                    for(int t:{-2,n+1}) if(int q=offset(p,d,t);q>=0) s.beyond.add(q);
                    if(n==5) {starts[d].add(p); fives.push_back(s);}
                    for(int i=0;i<n;++i) through[offset(p,d,i)][n-4].push_back(s);
                }
            }
        }
        if(fives.size()!=572) throw std::logic_error("15x15 window count");
    }
};
inline const Geometry geometry;
inline Mask completions(const Board& b,int side,bool exact) {
    Mask result,own=b.stones(side);
    for(int d=0;d<4;++d) {
        int s=directions[d][0]+16*directions[d][1];
        Mask starts=geometry.starts[d];
        if(exact) starts=starts & ~(own.left(s)|own.right(5*s));
        Mask a=own, c1=own.right(s), c2=own.right(2*s), c3=own.right(3*s), c4=own.right(4*s);
        Mask lo=a&c1, hi=c3&c4;
        result=result | (starts&c1&c2&hi) | (starts&a&c2&hi).left(s)
            | (starts&lo&hi).left(2*s) | (starts&lo&c2&c4).left(3*s)
            | (starts&lo&c2&c3).left(4*s);
    }
    return result & geometry.playable & ~b.occupied();
}
// Called when there is no immediate white completion. Every gain must belong
// to a black-free length-five window with exactly three existing white stones.
inline Mask whiteGains(const Board& b) {
    Mask result;
    for(int d=0;d<4;++d) {
        int s=directions[d][0]+16*directions[d][1];
        Mask start=geometry.starts[d];
        std::array<Mask,5> stone;
        for(int i=0;i<5;++i) {
            stone[i]=b.white.right(i*s);
            start=start & ~b.black.right(i*s);
        }
        for(int hole=0;hole<5;++hole) {
            auto& a=stone[(hole+1)%5];auto& c=stone[(hole+2)%5];
            auto& e=stone[(hole+3)%5];auto& f=stone[(hole+4)%5];
            Mask three=(a&c&(e|f)) | (e&f&(a|c));
            result=result | (start&three).left(hole*s);
        }
    }
    return result & geometry.playable & ~b.occupied();
}
enum BlackMove {Legal=0,ExactFive=1,Overline=2,DoubleFour=3,DoubleThree=4};
struct Three {Mask support,extensions;};
inline BlackMove blackMove(const Board& b,int p,Mask* trace=nullptr) {
    if(unsigned(p)>=Cells || b.occupied().has(p)) throw std::invalid_argument("expected empty black square");
    if(trace) *trace=*trace|geometry.rays[p];
    auto after=b.with(p,0);
    const Mask black=after.black, occupied=after.occupied();
    const auto& lines=geometry.through[p];
    for(const auto& s:lines[1])
        if(black.contains(s.body) && !(black&s.ends).any()) return ExactFive;
    for(const auto& s:lines[2]) if(black.contains(s.body)) return Overline;
    Mask firstFour;
    for(const auto& s:lines[1]) {
        if((s.body&b.white).any() || (s.ends&black).any()) continue;
        Mask support=s.body&black;
        if(support.count()!=4) continue;
        if(firstFour.any() && firstFour!=support) return DoubleFour;
        firstFour=support;
    }
    std::vector<Three> threes;
    for(const auto& s:lines[0]) {
        if(s.ends.count()!=2 || (s.ends&occupied).any() || (s.beyond&black).any() || (s.body&b.white).any()) continue;
        Mask support=s.body&black;
        if(support.count()!=3) continue;
        Mask ext=s.body&~support;
        auto it=std::find_if(threes.begin(),threes.end(),[&](auto& t){return t.support==support;});
        if(it==threes.end()) threes.push_back({support,ext});
        else it->extensions=it->extensions|ext;
    }
    int real=0;
    if(threes.size()>1) for(auto t:threes) {
        bool valid=false;
        while(t.extensions.any()) {
            if(blackMove(after,t.extensions.pop(),trace)==Legal) valid=true;
            if(valid && !trace) break;
        }
        real+=valid;
        if(real>=2 && !trace) return DoubleThree;
    }
    return real>=2?DoubleThree:Legal;
}
enum Leaf {Five=0,ForcedBlock=1,Fork=2,Foul=3};
struct Step {int gain,block,other; Leaf kind; BlackMove status=Legal;};
struct VcfLimits {
    uint64_t maxWhiteNodes=UINT64_MAX;
    bool (*cancel)(void*)=nullptr;
    void* context=nullptr;
};
struct VcfResult {bool win=false,exhausted=false; uint64_t nodes=0,hits=0; std::vector<Step> line;};
class Vcf {
    bool foul;
    VcfLimits limits;
    std::unordered_set<Board,BoardHash> failed;
    VcfResult out;
    bool defense(const Board& b,int gain,Mask targets) {
        if(!targets.any()) return false;
        int block=targets.pop();
        if(targets.any()) {
            out.line.push_back({gain,block,targets.pop(),Fork});return true;
        }
        if(foul) {
            BlackMove status=blackMove(b,block);
            if(status==ExactFive) return false;
            if(status!=Legal) {out.line.push_back({gain,block,-1,Foul,status});return true;}
        }
        out.line.push_back({gain,block,-1,ForcedBlock});
        if(attack(b.with(block,0))) return true;
        out.line.pop_back();return false;
    }
    bool attack(const Board& b,int chosen=-1) {
        if((limits.maxWhiteNodes!=UINT64_MAX && out.nodes>=limits.maxWhiteNodes)
            || (limits.cancel && limits.cancel(limits.context))) {
            out.exhausted=true;return false;
        }
        if(out.nodes<UINT64_MAX) ++out.nodes;
        if(chosen<0 && failed.contains(b)) {++out.hits;return false;}
        auto wins=completions(b,1,false);
        if(wins.any() && (chosen<0 || wins.has(chosen))) {
            out.line.push_back({chosen<0?wins.pop():chosen,-1,-1,Five});return true;
        }
        // A specified move may leave another already-existing white completion.
        // Inspect the resulting board literally for this diagnostic entry point.
        Mask danger=completions(b,0,true);
        if(chosen>=0) {
            if(danger.any() && (danger.count()!=1 || !danger.has(chosen))) return false;
            auto after=b.with(chosen,1);
            return defense(after,chosen,completions(after,1,false));
        }
        Mask gains;
        if(danger.count()<=1) {
            gains=whiteGains(b);
            if(danger.any()) gains=gains&danger;
        }
        while(gains.any()) {
            int gain=gains.pop();auto after=b.with(gain,1);
            if(defense(after,gain,completions(after,1,false))) return true;
            if(out.exhausted) return false;
        }
        failed.insert(b);
        return false;
    }
public:
    explicit Vcf(bool augmented,VcfLimits budget={}):foul(augmented),limits(budget) {}
    // Nonterminal, normal alternating Renju placement. Black-to-move is the
    // defense phase of an already outstanding four. Default limits are exhaustive.
    VcfResult solve(const Board& b,int side,int chosen=-1) {
        if(unsigned(side)>1 || chosen < -1 || (chosen>=0 && (side!=1 || chosen>=Cells || b.occupied().has(chosen))))
            throw std::invalid_argument("wrong turn or occupied gain");
        failed.clear();out={};
        if(limits.cancel && limits.cancel(limits.context)) {out.exhausted=true;return std::move(out);}
        out.win=side?attack(b,chosen):(!completions(b,0,true).any() && defense(b,-1,completions(b,1,false)));
        return std::move(out);
    }
};
} // namespace f15
