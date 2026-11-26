// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "certificates.h"
namespace f10 {
struct Clause {
    Points options; // encoded unordered pairs
    Points killedBy; // possible black preplay that alone invalidates this clause
};
inline int encode(int p,int q) {return std::min(p,q)*Cells+std::max(p,q);}
inline void uniqueSort(Points& p) {std::sort(p.begin(),p.end()); p.erase(std::unique(p.begin(),p.end()),p.end());}
inline bool intersects(const Points& a,const Points& b) {
    return std::any_of(a.begin(),a.end(),[&](int p){return contains(b,p);});
}
inline Points united(Points a,const Points& b) {a.insert(a.end(),b.begin(),b.end()); uniqueSort(a); return a;}
struct Shape {
    Points stones, holes, antiBlack; // four/three support, necessary empty sites, forbidden black flanks
};
struct Guard {
    std::vector<Clause> clauses;
    bool impossible=false;
    void add(const Points& need,const Points& holes,const Points& killed) {
        Clause row{{},killed};
        for (int p:need) {
            for (int q:need) if (p<q) row.options.push_back(encode(p,q));
            for (int q:holes) row.options.push_back(encode(p,q));
        }
        uniqueSort(row.options); uniqueSort(row.killedBy);
        if (row.options.empty() && row.killedBy.empty()) impossible=true;
        clauses.push_back(std::move(row));
    }
};
inline Guard buildGuard(const f8::Position& board,const Edges& white) {
    Guard guard;
    for (auto& e:white) {auto v=points(e); guard.add(v,{},v);}
    auto black=[&](int p){return !board.empty.has(p) && !board.white.has(p);};
    auto onlyFree=[&](Points v){std::erase_if(v,[&](int p){return !board.empty.has(p);});return v;};
    std::array<std::vector<Shape>,2> shapes;
    for (auto& dir:f8::dirs) for (int y=0;y<f8::N;++y) for (int x=0;x<f8::N;++x) {
        auto cell=[&](int t) {
            int u=x+t*dir[0],v=y+t*dir[1]; return f8::onBoard(u,v)?v*f8::N+u:NoMove;
        };
        if (cell(4)<0) continue;
        Points five; for (int t=0;t<5;++t) five.push_back(cell(t));
        if (std::any_of(five.begin(),five.end(),[&](int p){return board.white.has(p);})) continue;
        Points flanks; for (int p:{cell(-1),cell(5)}) if (p>=0) flanks.push_back(p);
        if (std::none_of(flanks.begin(),flanks.end(),black)) {
            for (int h:five) if (board.empty.has(h)) {
                Points stones=five; std::erase(stones,h);
                shapes[0].push_back({std::move(stones),{h},flanks});
            }
        }
        if (cell(5)<0 || board.white.has(cell(5))) continue;
        Points six=five; six.push_back(cell(5));
        guard.add(onlyFree(six),{},{});
        if (guard.impossible) return guard;
        if (!board.empty.has(cell(0)) || !board.empty.has(cell(5))) continue;
        Points outside; for (int p:{cell(-1),cell(6)}) if (p>=0) outside.push_back(p);
        if (std::any_of(outside.begin(),outside.end(),black)) continue;
        Points four{cell(1),cell(2),cell(3),cell(4)};
        for (int h:four) if (board.empty.has(h)) {
            Points stones=four; std::erase(stones,h);
            shapes[1].push_back({std::move(stones),{cell(0),h,cell(5)},outside});
        }
    }
    for (auto& family:shapes) {
        std::array<Points,Cells> at;
        for (int i=0;i<int(family.size());++i)
            for (int p:family[i].stones) if (board.empty.has(p)) at[p].push_back(i);
        for (int p=0;p<Cells;++p) for (int i=0;i<int(at[p].size());++i) for (int j=0;j<i;++j) {
            const auto& a=family[at[p][i]]; const auto& b=family[at[p][j]];
            if (a.stones==b.stones || intersects(a.stones,b.holes) || intersects(b.stones,a.holes) ||
                intersects(a.stones,b.antiBlack) || intersects(b.stones,a.antiBlack)) continue;
            // Two shapes need to meet at the newly placed black stone. Emit their union once.
            bool prior=false;
            for (int q:a.stones) if (q<p && board.empty.has(q) && contains(b.stones,q)) prior=true;
            if (prior) continue;
            auto holes=united(a.holes,b.holes);
            auto killed=united(holes,united(a.antiBlack,b.antiBlack));
            guard.add(onlyFree(united(a.stones,b.stones)),holes,onlyFree(killed));
        }
    }
    std::sort(guard.clauses.begin(),guard.clauses.end(),[](auto& a,auto& b){
        return a.options!=b.options?a.options<b.options:a.killedBy<b.killedBy;
    });
    guard.clauses.erase(std::unique(guard.clauses.begin(),guard.clauses.end(),[](auto& a,auto& b){
        return a.options==b.options && a.killedBy==b.killedBy;
    }),guard.clauses.end());
    return guard;
}
inline bool satisfy(const Guard& guard,int pre,const f8::CellsMask& empty,Proof& proof,int variant) {
    std::array<int,Cells> mate; mate.fill(NoMove);
    if (pre>=0) mate[pre]=pre;
    std::vector<const Clause*> open;
    for (auto& row:guard.clauses) if (!contains(row.killedBy,pre)) open.push_back(&row);
    // Only the current solver owns these counters. Reset the touched entries between calls.
    static thread_local std::array<int,Cells*Cells> frequency{};
    Points touched;
    for (auto row:open) for (int id:row->options) {
        if (id/Cells==pre || id%Cells==pre) continue;
        if (!frequency[id]++) touched.push_back(id);
    }
    struct Reset {
        Points& ids;
        ~Reset() {for (int id:ids) frequency[id]=0;}
    } reset{touched};
    while (!open.empty()) {
        const Clause* tight=nullptr; int fewest=INT_MAX;
        int keep=0;
        for (auto row:open) {
            int count=0; bool done=false;
            for (int id:row->options) {
                int a=id/Cells,b=id%Cells;
                if (mate[a]==b) {done=true;break;}
                count+=mate[a]<0 && mate[b]<0;
            }
            if (done) continue;
            if (!count) return false;
            if (count<fewest || (variant && count==fewest)) {tight=row; fewest=count;}
            open[keep++]=row;
        }
        open.resize(keep);
        if (!tight) break;
        int best=-1,bestScore=-1;
        for (int id:tight->options) {
            if (mate[id/Cells]>=0 || mate[id%Cells]>=0) continue;
            if (frequency[id]>bestScore || (variant && frequency[id]==bestScore)) {
                best=id;bestScore=frequency[id];
            }
        }
        int a=best/Cells,b=best%Cells; mate[a]=b; mate[b]=a;
    }
    int spare=-1; proof=Proof{}; proof.pre=pre; proof.mirror=true;
    for (int p=0;p<Cells;++p) if (empty.has(p) && mate[p]<0) {
        if (spare<0) spare=p;
        else {mate[p]=spare; mate[spare]=p; spare=-1;}
    }
    if (spare>=0) return false;
    for (int p=0;p<Cells;++p) if (mate[p]>p) proof.pairs.emplace_back(p,mate[p]);
    return true;
}
inline void defendBlack(const f8::Position& b,const Edges& white,bool first,Diagnosis& d) {
    if (!d.ordinary && !d.local) return;
    d.blanket=b.allFutureBlackMovesLegal();
    if (d.blanket) {d.accepted=true;return;}
    auto guard=buildGuard(b,white);
    if (guard.impossible) return;
    Points trials;
    auto trial=[&](int x){if (contains(d.pre,x) && !contains(trials,x)) trials.push_back(x);};
    if (first) {
        for (int x:d.t.one) trial(x);
        for (int x:d.t.robust) trial(x);
        if (d.proof.pre>=0) trial(d.proof.pre);
        for (int x:d.pre) if (x>=0) trial(x);
    } else trial(NoMove);
    for (int x:trials) {
        Proof p;
        if (satisfy(guard,x,b.empty,p,0) || satisfy(guard,x,b.empty,p,1)) {
            d.proof=std::move(p);d.accepted=true;return;
        }
    }
}
} // namespace f10
