// SPDX-License-Identifier: GPL-3.0-or-later
// F17. The padded bitboards, five-window kernel and recursive black adjudication
// derive from the local F15 implementation. C++20, single-threaded, no external libraries.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace f17 {
enum class Rule { Standard=1, Renju=2 };
enum class Side { Black=0, White=1 };
using Clock = std::chrono::steady_clock;
struct Limits {
    int plies=11;
    uint64_t nodes=UINT64_MAX;
    Clock::time_point deadline=Clock::time_point::max();
    bool liveThrees=true, rzone=true, foulTerminal=true, cache=true;
};
namespace detail {
struct Cutoff {};
struct Control {
    Clock::time_point until;
    void poll() const { if(Clock::now() >= until) throw Cutoff{}; }
};
inline Clock::time_point searchDeadline(Clock::time_point hard) {
    if(hard==Clock::time_point::max()) return hard;
    auto now=Clock::now();
    if(hard<=now) return hard;
    // Leave time for unwinding the AND tree and releasing its shared certificates.
    auto reserve=std::chrono::duration_cast<Clock::duration>(std::chrono::microseconds(100));
    return hard-std::min(reserve,(hard-now)/8);
}
}
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
    Board with(int p,int side) const {
        if(unsigned(p)>=Cells || unsigned(side)>1 || occupied().has(p)) throw std::invalid_argument("invalid placement");
        Board b=*this; (side?b.white:b.black).add(p);return b;
    }
};
// Omega's dense board uses 0 for empty, 1 for black and 2 for white.
inline bool fromCells(const uint8_t* cells,Board& out) {
    if(!cells) return false;
    Board b;
    for(int p=0;p<Cells;++p) {
        if(cells[p]>2) return false;
        if(cells[p]) (cells[p]==1?b.black:b.white).add(p);
    }
    out=b;
    return true;
}
namespace detail {
inline constexpr int directions[4][2]={{1,0},{0,1},{1,1},{-1,1}};
inline int offset(int p,int d,int t) {
    int x=p%Width+t*directions[d][0],y=p/Width+t*directions[d][1];
    return unsigned(x)<Width && unsigned(y)<Width? y*Width+x : -1;
}
struct Segment { Mask body,ends,beyond; };
struct Geometry {
    Mask playable;
    std::array<Mask,4> starts, openStarts;
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
                    if(n==4 && s.ends.count()==2) openStarts[d].add(p);
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
enum BlackMove {Legal=0,ExactFive=1,Overline=2,DoubleFour=3,DoubleThree=4};
struct Three {Mask support,extensions;};
inline BlackMove blackMove(const Board& b,int p,Mask* trace=nullptr,const Control* control=nullptr) {
    if(control) control->poll();
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
            if(blackMove(after,t.extensions.pop(),trace,control)==Legal) valid=true;
            if(valid && !trace) break;
        }
        real+=valid;
        if(real>=2 && !trace) return DoubleThree;
    }
    return real>=2?DoubleThree:Legal;
}

inline bool exact(Rule rule,int side) { return rule==Rule::Standard || side==0; }
inline bool wonHere(const Board& placed,int p,int side,Rule rule,Mask* reads=nullptr) {
    const auto own=placed.stones(side);
    for(const auto& line:geometry.through[p][1]) {
        if(own.contains(line.body) && (!exact(rule,side) || !(own&line.ends).any())) {
            if(reads) *reads=*reads|line.body|(exact(rule,side)?line.ends:Mask{});
            return true;
        }
    }
    return false;
}
inline bool terminal(const Board& b,Rule rule) {
    if(b.occupied()==geometry.playable) return true;
    for(int side=0;side<2;++side) {
        Mask own=b.stones(side);
        for(int d=0;d<4;++d) {
            int s=directions[d][0]+16*directions[d][1];
            if(rule==Rule::Renju && side==0 &&
               (geometry.starts[d]&own&own.right(s)&own.right(2*s)&own.right(3*s)&own.right(4*s)&own.right(5*s)).any())
                return true;
            Mask ends=exact(rule,side)?own.left(s)|own.right(5*s):Mask{};
            if((geometry.starts[d]&~ends&own&own.right(s)&own.right(2*s)&own.right(3*s)&own.right(4*s)).any())
                return true;
        }
    }
    return false;
}
inline bool forbidden(const Board& b,int p,int side,Rule r,Mask* reads,const Control* c) {
    return r==Rule::Renju && side==0 && blackMove(b,p,reads,c)>ExactFive;
}
// A five-window with exactly three own stones has two possible gains. Starts
// eliminate walls, opponent stones, and endpoints that would yield an overline.
inline Mask fourGains(const Board& b,int side,Rule r) {
    const auto own=b.stones(side), opp=b.stones(1-side);
    Mask gains;
    for(int d=0;d<4;++d) {
        int s=directions[d][0]+16*directions[d][1];
        Mask start=geometry.starts[d];
        if(exact(r,side)) start=start & ~(own.left(s)|own.right(5*s));
        std::array<Mask,5> stone;
        for(int i=0;i<5;++i) {
            stone[i]=own.right(i*s);
            start=start & ~opp.right(i*s);
        }
        for(int hole=0;hole<5;++hole) {
            const auto &a=stone[(hole+1)%5], &c=stone[(hole+2)%5];
            const auto &e=stone[(hole+3)%5], &f=stone[(hole+4)%5];
            const auto three=(a&c&(e|f)) | (e&f&(a|c));
            gains=gains|(start&three).left(hole*s);
        }
    }
    return gains & geometry.playable & ~b.occupied();
}
inline Mask threeGains(const Board& b,int side,Rule r,Mask& multipleAxes) {
    const auto own=b.stones(side), opp=b.stones(1-side), occ=b.occupied();
    Mask gains;
    for(int d=0;d<4;++d) {
        int s=directions[d][0]+16*directions[d][1];
        Mask start=geometry.openStarts[d] & ~(occ.left(s)|occ.right(4*s));
        if(exact(r,side)) start=start & ~(own.left(2*s)|own.right(5*s));
        std::array<Mask,4> stone;
        for(int i=0;i<4;++i) {
            stone[i]=own.right(i*s);
            start=start & ~opp.right(i*s);
        }
        Mask axis;
        for(int hole=0;hole<4;++hole) {
            const auto &a=stone[(hole+1)%4], &c=stone[(hole+2)%4], &e=stone[(hole+3)%4];
            axis=axis|(start&((a&c)|(a&e)|(c&e))).left(hole*s);
        }
        axis=axis & geometry.playable & ~occ;
        multipleAxes=multipleAxes|(axis&gains);
        gains=gains|axis;
    }
    return gains;
}
inline Mask anchoredFour(const Board& after,int gain,int side,Rule r,Mask* reads=nullptr) {
    Mask wins;
    const Mask own=after.stones(side), opp=after.stones(1-side);
    for(const auto& s:geometry.through[gain][1]) {
        if((opp&s.body).any() || (exact(r,side) && (own&s.ends).any())) continue;
        Mask gaps=s.body&~own;
        if(gaps.count()!=1) continue;
        wins=wins|gaps;
        if(reads) *reads=*reads|s.body|(exact(r,side)?s.ends:Mask{});
    }
    return wins;
}
inline Mask dependencies(const Board& b,int side,Rule r,int previous,bool live) {
    if(previous<0) return {};
    Mask gains;
    const auto own=b.stones(side), opp=b.stones(1-side), occ=b.occupied();
    for(const auto& s:geometry.through[previous][live?0:1]) {
        if((s.body&opp).any() || (s.body&own).count()!=(live?2:3)) continue;
        if(live && (s.ends.count()!=2 || (s.ends&occ).any())) continue;
        if(exact(r,side) && (own&(live?s.beyond:s.ends)).any()) continue;
        gains=gains|(s.body&~occ);
    }
    return gains;
}
} // namespace detail

struct Extension {
    int gain=-1, left=-1, right=-1;
    Mask cost, support;
};
enum class Kind { Five, Fork, ForcedFour, FoulFour, LiveThree };
struct Proof;
using Certificate=std::shared_ptr<const Proof>;
struct Edge {int reply; Certificate next;};
struct Proof {
    Kind kind=Kind::Five;
    int gain=-1, block=-1, other=-1, plies=1;
    bool usesThree=false;
    Mask support;
    std::vector<Extension> extensions;
    std::vector<Edge> edges;
};
struct Threat {
    int gain;
    bool five=false, foulFour=false;
    Mask completions;
    std::vector<Extension> extensions;
};
namespace detail {
inline std::vector<Extension> extensions(const Board& after,int gain,int side,Rule rule,const Control* ctl) {
    std::vector<Extension> found;
    const auto own=after.stones(side), opp=after.stones(1-side), occ=after.occupied();
    for(const auto& s:geometry.through[gain][0]) {
        if(s.ends.count()!=2 || (s.ends&occ).any() || (s.body&opp).any()) continue;
        if(exact(rule,side) && (own&s.beyond).any()) continue;
        Mask gap=s.body&~own;
        if(gap.count()!=1) continue;
        int e=gap.pop();
        Mask trace;
        if(side==0 && rule==Rule::Renju) {
            if(blackMove(after,e,&trace,ctl)!=Legal) continue;
        } else if(wonHere(after.with(e,side),e,side,rule)) continue;
        Mask ends=s.ends;
        Extension w;
        w.gain=e; w.left=ends.pop(); w.right=ends.pop();
        w.cost=trace|s.ends; w.cost.add(e);
        w.support=w.cost|s.body|(exact(rule,side)?s.beyond:Mask{});
        found.push_back(w);
    }
    return found;
}
inline Mask localCosts(const std::vector<Extension>& ext) {
    Mask critical=geometry.playable;
    for(const auto& e:ext) critical=critical&e.cost;
    return critical;
}
inline bool usable(const Board& afterReply,const Extension& e,int side,Rule rule,const Control* ctl) {
    const auto occ=afterReply.occupied();
    if(occ.has(e.gain) || occ.has(e.left) || occ.has(e.right)) return false;
    if(forbidden(afterReply,e.gain,side,rule,nullptr,ctl)) return false;
    const auto after=afterReply.with(e.gain,side);
    if(wonHere(after,e.gain,side,rule)) return true;
    if(completions(after,1-side,exact(rule,1-side)).any()) return false;
    return wonHere(after.with(e.left,side),e.left,side,rule)
        && wonHere(after.with(e.right,side),e.right,side,rule);
}
} // namespace detail

// Exact, unbudgeted inspection of threats newly made through the gain. A legal
// guard may remove a geometric candidate. A square may create a four and a three.
inline std::vector<Threat> threats(const Board& b,Side attacker,Rule rule) {
    using namespace detail;
    int side=int(attacker);
    if(unsigned(side)>1 || (rule!=Rule::Renju && rule!=Rule::Standard)
       || (b.occupied()&~geometry.playable).any() || (b.black&b.white).any() || terminal(b,rule)) return {};
    Mask multiple;
    Mask immediate=completions(b,side,exact(rule,side));
    Mask moves=immediate|fourGains(b,side,rule)|threeGains(b,side,rule,multiple);
    std::vector<Threat> result;
    while(moves.any()) {
        int m=moves.pop();
        if(forbidden(b,m,side,rule,nullptr,nullptr)) continue;
        auto after=b.with(m,side);
        Threat t{m};
        t.five=immediate.has(m);
        if(!t.five) {
            t.completions=anchoredFour(after,m,side,rule);
            t.extensions=extensions(after,m,side,rule,nullptr);
            if(rule==Rule::Renju && side==1 && t.completions.count()==1) {
                Mask target=t.completions;
                t.foulFour=blackMove(after,target.pop())>ExactFive
                    && !completions(after,0,true).any();
            }
        }
        if(t.five || t.completions.any() || !t.extensions.empty()) result.push_back(std::move(t));
    }
    return result;
}

struct Statistics {
    uint64_t orNodes=0, andNodes=0, replies=0, outside=0, counterCandidates=0;
    uint64_t cacheProbes=0, cacheHits=0, differentBoardHits=0, guardRejects=0;
    uint64_t foulLeaves=0, dependentGains=0;
    uint64_t nodes() const { return orNodes+andNodes; }
};
enum class Status { Proven, Unknown, Terminal, Invalid };
struct Result {
    Status status=Status::Unknown;
    Certificate proof;
    Statistics stats;
    bool exhausted=false;
};
namespace detail {
class Search {
    const Rule rule;
    const int side, defender;
    const Limits limits;
    Control control;
    Statistics stats{};
    struct Entry {Board original; Certificate proof;};
    std::array<Entry,32> table{};
    unsigned next=0, count=0;
    void node(bool attack) {
        control.poll();
        if(stats.nodes()>=limits.nodes) throw Cutoff{};
        if(attack) ++stats.orNodes; else ++stats.andNodes;
    }
    bool counterwin(const Board& b) const {return completions(b,defender,exact(rule,defender)).any();}
public:
    // Replay the cached strategy on the new diagram. A remote four may create
    // an entirely new compulsory reply at a live-three frame. The caller must
    // have matched the colored/empty projection of the original certificate.
    bool guarded(const Board& b,const Proof& p) {
        control.poll();
        if(b.occupied().has(p.gain) || forbidden(b,p.gain,side,rule,nullptr,&control)) return false;
        auto a=b.with(p.gain,side);
        if(p.kind==Kind::Five) return wonHere(a,p.gain,side,rule);
        if(wonHere(a,p.gain,side,rule) || counterwin(a)) return false;
        if(p.kind!=Kind::LiveThree) {
            auto wins=completions(a,side,exact(rule,side));
            if(!wins.has(p.block)) return false;
            if(p.kind==Kind::Fork) return wins.has(p.other);
            bool foul=forbidden(a,p.block,defender,rule,nullptr,&control);
            if(p.kind==Kind::FoulFour) return foul;
            return !foul && guarded(a.with(p.block,defender),*p.edges.front().next);
        }
        auto needed=(localCosts(p.extensions)|fourGains(a,defender,rule));
        for(const auto& edge:p.edges) needed.add(edge.reply);
        needed=needed&~a.occupied();
        while(needed.any()) {
            control.poll();
            int d=needed.pop();
            if(forbidden(a,d,defender,rule,nullptr,&control)) continue;
            auto parried=a.with(d,defender);
            auto edge=std::find_if(p.edges.begin(),p.edges.end(),[d](const Edge& x){return x.reply==d;});
            if(edge!=p.edges.end()) {
                if(!guarded(parried,*edge->next)) return false;
            } else if(std::none_of(p.extensions.begin(),p.extensions.end(),
                    [&](const auto& e){return usable(parried,e,side,rule,&control);})) return false;
        }
        return true;
    }
private:
    Certificate probe(const Board& b,int plies,bool allowLive) {
        if(!limits.cache || !count) return {};
        ++stats.cacheProbes;
        for(unsigned age=0;age<count;++age) {
            const auto& e=table[(next+table.size()-1-age)%table.size()];
            const auto& p=*e.proof;
            if(p.plies>plies || (p.usesThree && !allowLive)) continue;
            if((e.original.black&p.support)!=(b.black&p.support)
                || (e.original.white&p.support)!=(b.white&p.support)) continue;
            if(e.original==b) {++stats.cacheHits; return e.proof;}
            if(guarded(b,p)) {++stats.cacheHits; ++stats.differentBoardHits; return e.proof;}
            ++stats.guardRejects;
        }
        return {};
    }
    Certificate remember(const Board& b,Certificate p) {
        if(limits.cache && p) {
            table[next]={b,p};
            next=(next+1)%table.size();
            if(count<table.size()) ++count;
        }
        return p;
    }
    Certificate four(const Board& b,int gain,int plies,bool allowLive) {
        node(false);
        Mask support;
        if(forbidden(b,gain,side,rule,&support,&control)) return {};
        auto a=b.with(gain,side);
        auto targets=anchoredFour(a,gain,side,rule,&support);
        if(!targets.any()) return {};
        int block=targets.pop(), other=-1;
        Kind kind;
        Certificate child;
        if(targets.any()) {kind=Kind::Fork; other=targets.pop();}
        else if(forbidden(a,block,defender,rule,&support,&control)) {
            if(!limits.foulTerminal) return {};
            kind=Kind::FoulFour;
            ++stats.foulLeaves;
        } else {
            kind=Kind::ForcedFour;
            ++stats.replies;
            child=attack(a.with(block,defender),plies-2,allowLive,gain);
            if(!child) return {};
        }
        auto p=std::make_shared<Proof>();
        p->kind=kind; p->gain=gain; p->block=block; p->other=other;
        p->support=support; p->plies=3;
        if(child) {
            p->edges.push_back({block,child}); p->support=p->support|child->support;
            p->plies=2+child->plies; p->usesThree=child->usesThree;
        }
        return p;
    }
    Certificate three(const Board& b,int gain,int plies) {
        node(false);
        Mask support;
        support.add(gain);
        if(forbidden(b,gain,side,rule,&support,&control)) return {};
        auto a=b.with(gain,side);
        auto ext=extensions(a,gain,side,rule,&control);
        if(ext.empty()) return {};
        auto counters=fourGains(a,defender,rule);
        Mask empties=geometry.playable&~a.occupied();
        Mask needed=(localCosts(ext)|counters)&empties;
        stats.counterCandidates+=counters.count();
        if(limits.rzone) stats.outside+=(empties&~needed).count();
        else needed=empties;
        // Counterchecks first. A single surviving line refutes this OR gain.
        std::array<Mask,2> groups={needed&counters,needed&~counters};
        std::vector<Edge> edges;
        int length=5;
        for(auto group:groups) while(group.any()) {
            control.poll();
            int d=group.pop();
            if(forbidden(a,d,defender,rule,nullptr,&control)) continue;
            ++stats.replies;
            auto parried=a.with(d,defender);
            // This visits the OR response explicitly. Only F3 can discharge
            // disjoint replies collectively without visiting this node.
            node(true);
            if(std::any_of(ext.begin(),ext.end(),[&](const auto& e){
                return usable(parried,e,side,rule,&control);
            })) continue;
            auto child=attack(parried,plies-2,true,gain,false);
            if(!child) return {};
            length=std::max(length,2+child->plies);
            support=support|child->support; support.add(d);
            edges.push_back({d,child});
        }
        for(const auto& e:ext) support=support|e.support;
        auto p=std::make_shared<Proof>();
        p->kind=Kind::LiveThree; p->gain=gain; p->plies=length; p->usesThree=true;
        p->extensions=std::move(ext); p->edges=std::move(edges); p->support=support;
        return p;
    }
    Certificate attack(const Board& b,int plies,bool allowLive,int previous,bool charge=true) {
        if(charge) node(true);
        auto wins=completions(b,side,exact(rule,side));
        if(wins.any()) {
            auto p=std::make_shared<Proof>();
            p->gain=wins.pop();
            wonHere(b.with(p->gain,side),p->gain,side,rule,&p->support);
            return p;
        }
        if(plies<3) return {};
        auto danger=completions(b,defender,exact(rule,defender));
        if(danger.count()>1) return {};
        if(auto saved=probe(b,plies,allowLive)) return saved;
        auto filter=danger.any()?danger:geometry.playable;
        auto fours=fourGains(b,side,rule)&filter;
        auto dep=dependencies(b,side,rule,previous,false)&fours;
        for(auto group:{dep,fours&~dep}) while(group.any()) {
            int gain=group.pop();
            if(dep.has(gain)) ++stats.dependentGains;
            if(auto p=four(b,gain,plies,allowLive)) return remember(b,p);
        }
        if(!allowLive || plies<5) return {};
        Mask multi;
        // Four gains have already had their forcing single-block branch searched.
        // With the F15 terminal disabled its forbidden sole block remains unresolved.
        auto threes=threeGains(b,side,rule,multi)&filter&~fours;
        dep=dependencies(b,side,rule,previous,true)&threes;
        multi=multi&threes;
        for(auto group:{multi,dep&~multi,threes&~(dep|multi)}) while(group.any()) {
            int gain=group.pop();
            if(dep.has(gain)) ++stats.dependentGains;
            if(auto p=three(b,gain,plies)) return remember(b,p);
        }
        return {};
    }
public:
    Search(Rule r,Side s,const Limits& l):rule(r),side(int(s)),defender(1-side),limits(l),control{searchDeadline(l.deadline)} {}
    Result run(const Board& b) {
        Result out;
        try {
            control.poll();
            if(unsigned(side)>1 || (rule!=Rule::Renju && rule!=Rule::Standard)
                || limits.plies<1 || limits.plies>225 || (b.black&b.white).any()
                || (b.occupied()&~geometry.playable).any()) {out.status=Status::Invalid; return out;}
            if(terminal(b,rule)) {out.status=Status::Terminal; return out;}
            const int h=std::min(limits.plies,Cells-b.occupied().count());
            // Continuous fours first. Iterative live-three rounds keep shallow
            // forcing combinations ahead of speculative eleven-ply branches.
            auto p=attack(b,h,false,-1);
            if(!p && limits.liveThrees)
                for(int d=5;d<=h && !p;d+=2) p=attack(b,d,true,-1);
            control.poll();
            if(p) {out.proof=std::move(p); out.status=Status::Proven;}
        } catch(const Cutoff&) {out.exhausted=true;}
        out.stats=stats;
        return out;
    }
};
} // namespace detail

// Standard and Renju, normal 15x15 placement. Use an explicit attacker for
// hypothetical pass queries. Cache lifetime ends with this call.
inline Result solve(const Board& board,Side attacker,Rule rule,Limits limits={}) {
    auto result=detail::Search(rule,attacker,limits).run(board);
    if(result.proof && Clock::now()>=limits.deadline) {
        result.proof.reset();
        result.status=Status::Unknown;
        result.exhausted=true;
    }
    return result;
}
} // namespace f17
