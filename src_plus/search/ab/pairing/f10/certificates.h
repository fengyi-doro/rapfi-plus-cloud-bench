// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "../prover.h"
#include <climits>
#include <numeric>
#include <vector>

namespace f10 {
using f8::Cells;
using f8::Edge;
using Edges = std::vector<Edge>;
using Pair = std::pair<int,int>;
using Pairs = std::vector<Pair>;
using Points = std::vector<int>;
constexpr int NoMove = -1;
constexpr int LocalVertices = 20;
constexpr int LocalEdges = 128;

inline bool has(const Edge& e,int p) {
    return std::find(e.p.begin(),e.p.begin()+e.n,p)!=e.p.begin()+e.n;
}
inline bool subset(const Edge& a,const Edge& b) {
    for (int i=0;i<a.n;++i) if (!has(b,a.p[i])) return false;
    return true;
}
inline Points points(const Edge& e) { return {e.p.begin(),e.p.begin()+e.n}; }
inline Edges minimal(Edges edges) {
    std::stable_sort(edges.begin(),edges.end(),[](auto& a,auto& b){return a.n<b.n;});
    std::array<Points,Cells> byFirst;
    Edges out;
    for (auto& e:edges) {
        if (!e.n) return {e};
        bool redundant=false;
        for (int i=0;i<e.n && !redundant;++i)
            for (int j:byFirst[e.p[i]]) if (subset(out[j],e)) { redundant=true; break; }
        if (!redundant) { byFirst[e.p[0]].push_back(int(out.size())); out.push_back(e); }
    }
    return out;
}
inline Edges residual(const Edges& edges,int breaker,int maker=NoMove) {
    Edges result;
    for (auto e:edges) if (!has(e,breaker)) {
        auto end=std::remove(e.p.begin(),e.p.begin()+e.n,maker);
        e.n=uint8_t(end-e.p.begin()); result.push_back(e);
    }
    return result;
}
inline Points active(const Edges& edges) {
    std::array<bool,Cells> seen{}; Points out;
    for (auto& e:edges) for (int i=0;i<e.n;++i) {
        int p=e.p[i]; if (!seen[p]) { seen[p]=true; out.push_back(p); }
    }
    return out;
}
inline bool contains(const Points& v,int p) {return std::find(v.begin(),v.end(),p)!=v.end();}
inline Points commonAfter(const Edges& es,int first) {
    Points common; bool init=false;
    for (auto& e:es) if (!has(e,first)) {
        if (!init) { common=points(e); init=true; }
        else std::erase_if(common,[&](int p){return !has(e,p);});
        if (common.empty()) break;
    }
    return common;
}
struct Transversal { int tau=3; Points one, robust; }; // 3 means greater than two
inline Transversal transversal(const Edges& es) {
    Transversal t;
    if (es.empty()) { t.tau=0; return t; }
    t.one=commonAfter(es,NoMove);
    if (!t.one.empty()) { t.tau=1; return t; }
    for (int x:points(es[0])) if (!commonAfter(es,x).empty()) { t.tau=2; break; }
    if (t.tau==2) for (int x:active(es))
        if (commonAfter(es,x).size()>=2) t.robust.push_back(x);
    return t;
}
struct Branch { int attack, reply; Pairs pairs; };
struct Block { bool potential; Edges protect; std::vector<Branch> branches; };
struct Proof {
    int pre=NoMove;
    Pairs pairs;
    std::vector<Block> blocks;
    bool mirror=false;
    int peeled=0, degree2=0, degree3=0;
};
struct Preferences {
    std::array<int,3601> freq{};
    std::array<int,Cells> degree{};
    int twicePairCapacity=0;
    explicit Preferences(const Edges& edges) {
        std::array<int,Cells> maxFreq{};
        for (auto& e:edges) for (int i=0;i<e.n;++i) {
            int p=e.p[i]; ++degree[p];
            for (int j=0;j<i;++j) {
                int q=e.p[j], id=f8::geometry.pairId[p][q];
                int n=++freq[id];
                maxFreq[p]=std::max(maxFreq[p],n); maxFreq[q]=std::max(maxFreq[q],n);
            }
        }
        twicePairCapacity=std::accumulate(maxFreq.begin(),maxFreq.end(),0);
    }
};
// Multiple windows can use the same pair. Two deterministic tie orders.
inline bool pairCover(const Edges& edges,int pre,const Preferences& pref,Pairs& out,int variant=0) {
    int hit=pre<0?0:pref.degree[pre];
    if (2*(int(edges.size())-hit)>pref.twicePairCapacity) return false;
    std::array<int,Cells> mate; mate.fill(-1); out.clear();
    for (auto& e:edges) {
        if (has(e,pre)) continue;
        int best=-100000, a=-1,b=-1;
        bool covered=false;
        for (int i=0;i<e.n && !covered;++i) {
            int p=e.p[i];
            for (int j=0;j<i;++j) {
                int q=e.p[j];
                if (mate[p]==q) { covered=true; break; }
                if (mate[p]>=0 || mate[q]>=0) continue;
                int reuse=pref.freq[f8::geometry.pairId[p][q]];
                int cost=pref.degree[p]+pref.degree[q]+2*std::max(pref.degree[p],pref.degree[q]);
                int score=(variant?12:256)*reuse-cost;
                if (score>best || (variant && score==best)) {best=score; a=p; b=q;}
            }
        }
        if (covered) continue;
        if (a<0) {out.clear(); return false;}
        mate[a]=b; mate[b]=a; out.emplace_back(a,b);
    }
    return true;
}
// Necessary restrictions from immediate wins and two distinct two-edges at a vertex.
inline Points preplays(const Edges& es,bool first,const f8::CellsMask& empty,const Preferences& pref) {
    Points pre{NoMove};
    if (first) for (int p=0;p<Cells;++p) if (empty.has(p)) pre.push_back(p);
    std::array<Points,Cells> neighbors;
    for (auto& e:es) {
        if (e.n>2 || pre.empty()) break;
        if (e.n==1) std::erase_if(pre,[&](int x){return x!=e.p[0];});
        if (e.n==2) {
            for (int k=0;k<2;++k) {
                int c=e.p[k], b=e.p[1-k];
                for (int a:neighbors[c])
                    std::erase_if(pre,[&](int x){return x!=a && x!=b && x!=c;});
                neighbors[c].push_back(b);
            }
        }
    }
    std::stable_sort(pre.begin(),pre.end(),[&](int a,int b){
        return (a<0?1000:pref.degree[a])>(b<0?1000:pref.degree[b]);
    });
    return pre;
}
inline bool potential(const Edges& edges) {
    std::array<int,Cells> risk{}; int total=0, largest=0;
    for (auto& e:edges) {
        int w=32>>e.n; total+=w;
        for (int i=0;i<e.n;++i) largest=std::max(largest,risk[e.p[i]]+=w);
        if (total+largest>=32) return false;
    }
    return true;
}
inline std::vector<Edges> components(const Edges& edges) {
    std::array<int,Cells> parent;
    std::iota(parent.begin(),parent.end(),0);
    auto find=[&](int v) {
        while (parent[v]!=v) {parent[v]=parent[parent[v]]; v=parent[v];} return v;
    };
    for (auto& e:edges) for (int i=1;i<e.n;++i) parent[find(e.p[i])]=find(e.p[0]);
    std::array<int,Cells> id; id.fill(-1);
    std::vector<Edges> result;
    for (auto& e:edges) {
        int root=find(e.p[0]);
        if (id[root]<0) {id[root]=int(result.size()); result.emplace_back();}
        result[id[root]].push_back(e);
    }
    return result;
}
// Two endpoints confined to the same 1, 2 or 3 rows block every one of those rows.
// Repeating this also peels loose chains from their ends.
inline Edges peel(Edges es,Proof& proof) {
    for (;;) {
        std::array<Points,Cells> incident;
        for (int k=0;k<int(es.size());++k)
            for (int p:points(es[k])) incident[p].push_back(k);
        std::vector<bool> done(es.size());
        bool changed=false;
        for (int k=0;k<int(es.size());++k) {
            if (done[k]) continue;
            auto& e=es[k]; bool found=false;
            for (int i=0;i<e.n && !found;++i) for (int j=0;j<i;++j) {
                int a=e.p[i], b=e.p[j], d=int(incident[a].size());
                if (d>3 || incident[a]!=incident[b]) continue;
                if (std::any_of(incident[a].begin(),incident[a].end(),[&](int z){return done[z];})) continue;
                proof.pairs.emplace_back(a,b); ++proof.peeled;
                proof.degree2+=d==2; proof.degree3+=d==3;
                for (int z:incident[a]) done[z]=true;
                found=changed=true; break;
            }
        }
        if (!changed) return es;
        Edges remaining;
        for (int i=0;i<int(es.size());++i) if (!done[i]) remaining.push_back(es[i]);
        es.swap(remaining);
    }
}
inline bool flexible(const Edges& es,Block& result) {
    Points vertices=active(es);
    if (int(vertices.size())>LocalVertices) return false;
    for (int m:vertices) {
        auto after=minimal(residual(es,NoMove,m));
        if (!after.empty() && !after.front().n) return false;
        Preferences pref(after);
        int forced=NoMove;
        for (auto& e:after) {
            if (e.n!=1) break;
            if (forced>=0 && forced!=e.p[0]) return false;
            forced=e.p[0];
        }
        bool proved=false;
        for (int b:vertices) {
            if (b==m || (forced>=0 && forced!=b)) continue;
            Pairs pairs;
            if (pairCover(after,b,pref,pairs) || pairCover(after,b,pref,pairs,1)) {
                result.branches.push_back({m,b,std::move(pairs)}); proved=true; break;
            }
        }
        if (!proved) return false;
    }
    return true;
}
inline bool decompose(const Edges& es,int pre,Proof& proof) {
    auto left=peel(residual(es,pre),proof);
    for (auto& group:components(left)) {
        Preferences pref(group); Pairs pairs;
        if (pairCover(group,NoMove,pref,pairs) || pairCover(group,NoMove,pref,pairs,1)) {
            proof.pairs.insert(proof.pairs.end(),pairs.begin(),pairs.end()); continue;
        }
        Block block{potential(group),group,{}};
        if (!block.potential && !flexible(group,block)) return false;
        proof.blocks.push_back(std::move(block));
    }
    return true;
}
#include "reduction.inc"
struct Diagnosis {
    Transversal t;
    Points pre;
    bool ordinary=false, local=false, accepted=false, blanket=false;
    Proof proof;
};
inline Diagnosis unrestricted(const Edges& es,bool first,const f8::CellsMask& empty) {
    Diagnosis d; d.t=transversal(es);
    Preferences pref(es);
    d.pre=preplays(es,first,empty,pref);
    if (first && !d.t.one.empty()) {
        d.ordinary=true; d.proof.pre=d.t.one.front(); return d;
    }
    if (first && !d.t.robust.empty()) {
        d.ordinary=true; d.proof.pre=d.t.robust.front();
        auto left=commonAfter(es,d.proof.pre);
        d.proof.pairs.emplace_back(left[0],left[1]); return d;
    }
    for (int x:d.pre) {
        if (x>=0 && !pref.degree[x]) continue; // identical to NoMove for unrestricted defence
        if (pairCover(es,x,pref,d.proof.pairs) || pairCover(es,x,pref,d.proof.pairs,1)) {
            d.proof.pre=x; d.ordinary=true; return d;
        }
    }
    if (int(es.size())<=LocalEdges) for (int x:d.pre) {
        int hit=x<0?0:pref.degree[x];
        if ((x>=0 && !hit) || 2*(int(es.size())-hit)>pref.twicePairCapacity) continue;
        if (reducedPairing(residual(es,x),d.proof.pairs)) {
            d.proof.pre=x;d.ordinary=true;return d;
        }
    }
    if (int(es.size())<=LocalEdges) for (int x:d.pre) {
        if (x>=0 && !pref.degree[x]) continue;
        Proof p; p.pre=x;
        if (decompose(es,x,p)) {d.local=true; d.proof=std::move(p); return d;}
        p=Proof{}; p.pre=x;
        if (projectedLocal(es,x,p)) {d.local=true; d.proof=std::move(p); return d;}
    }
    return d;
}
} // namespace f10
