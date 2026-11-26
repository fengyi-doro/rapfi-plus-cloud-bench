// Two-slot SDR after forced-pair projection, followed by shared-pair choices.
// An edge containing an assigned pair is protected. Other edges lose used endpoints.
inline Edges project(const Edges& es,const Pairs& chosen) {
    std::array<int,Cells> mate; mate.fill(NoMove);
    for (auto [a,b]:chosen) {mate[a]=b;mate[b]=a;}
    Edges left;
    for (auto& e:es) {
        Edge r; bool covered=false;
        for (int i=0;i<e.n;++i) {
            int v=e.p[i];
            if (mate[v]<0) r.p[r.n++]=v;
            else if (has(e,mate[v])) {covered=true;break;}
        }
        if (!covered) left.push_back(r);
    }
    return minimal(std::move(left));
}
inline bool slots(const Edges& es,Pairs& certificate) {
    if (es.empty()) return true;
    if (2*es.size()>active(es).size()) return false;
    std::array<int,Cells> owner;owner.fill(NoMove);
    std::array<int,Cells> visited{};
    Points image(2*es.size(),NoMove);
    for (int slot=0;slot<int(image.size());++slot) {
        auto augment=[&](auto& self,int u)->bool {
            const auto& e=es[u/2];
            for (int i=0;i<e.n;++i) {
                int p=e.p[i];
                if (visited[p]==slot+1) continue;
                visited[p]=slot+1;
                if (owner[p]<0 || self(self,owner[p])) {
                    owner[p]=u;image[u]=p;return true;
                }
            }
            return false;
        };
        if (!augment(augment,slot)) return false;
    }
    for (int i=0;i<int(image.size());i+=2) certificate.emplace_back(image[i],image[i+1]);
    return true;
}
inline bool reducedPairing(Edges left,Pairs& certificate) {
    certificate.clear();
    while (!left.empty()) {
        if (left.front().n<2) return false;
        Pairs add;
        if (left.front().n==2) {
            std::array<bool,Cells> busy{};
            for (auto& e:left) {
                if (e.n!=2) break;
                int a=e.p[0],b=e.p[1];
                if (busy[a] || busy[b]) return false;
                busy[a]=busy[b]=true;add.emplace_back(a,b);
            }
        } else {
            if (slots(left,certificate)) return true;
            Preferences pref(left);
            int score=INT_MIN; Pair choice;
            for (auto& e:left) for (int i=0;i<e.n;++i) for (int j=0;j<i;++j) {
                int a=e.p[i],b=e.p[j],freq=pref.freq[f8::geometry.pairId[a][b]];
                if (freq<=1) continue;
                int cost=256*freq-pref.degree[a]-pref.degree[b];
                if (cost>score) {score=cost;choice={a,b};}
            }
            if (score==INT_MIN) return false;
            add.push_back(choice);
        }
        certificate.insert(certificate.end(),add.begin(),add.end());
        left=project(left,add);
    }
    return true;
}
// Use the forced short pairs as an independent subsystem. Shortened residual
// windows may now split into multiple disjoint adaptive components.
inline bool projectedLocal(const Edges& es,int pre,Proof& p) {
    auto rows=residual(es,pre);
    if (rows.empty() || rows.front().n!=2) return false;
    std::array<bool,Cells> used{};
    for (auto& e:rows) {
        if (e.n!=2) break;
        int a=e.p[0],b=e.p[1];
        if (used[a] || used[b]) return false;
        used[a]=used[b]=true;p.pairs.emplace_back(a,b);
    }
    auto left=project(rows,p.pairs);
    if (!left.empty() && left.front().n<2) return false;
    return decompose(left,NoMove,p);
}
