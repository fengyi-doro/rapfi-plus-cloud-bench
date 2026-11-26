// Check every precomputed move against a destination-first geometric specification.
#include "eval/mix9svqnnue.cpp"
#include <array>
#include <algorithm>
#include <cstdint>
#include <iostream>

int main() {
    using namespace Evaluation::mix9svq;
    uint64_t moves=0, destinations=0, changed=0;
    for (int n=5;n<=22;++n) {
        std::array<int8_t,32> group{};
        const int one=n/3+(n%3==2), two=2*(n/3)+(n%3!=0);
        for (int x=0;x<n;++x) group[x]=(x>=one)+(x>=two);
        const auto &walks=geometryFor(n,group.data());
        for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
            const auto &walk=walks[y*n+x];
            std::array<uint8_t,22*22> affected{}, mask{};
            unsigned centres=0;
            for (int i=0;i<walk.pointCount;++i) {
                const auto &p=walk.points[i];
                const int py=p.inner/n, px=p.inner%n;
                if (p.outer!=py*(n+2)+px || p.group!=group[py]*3+group[px]) return 1;
#ifdef RAPFI_V4_SCATTER
                for (int tap=0;tap<9;++tap) {
                    const int tx=px+tap%3-1, ty=py+tap/3-1;
                    unsigned slot=walk.scatter[i][tap];
                    if (tx<0 || ty<0 || tx>=n || ty>=n) {
                        if (slot!=walk.groupEnds.back()) return 8;
                    } else if (slot>=walk.groupEnds.back() ||
                               walk.targets[slot]!=(ty+1)*(n+2)+tx+1) return 9;
                }
#endif
                if (mask[p.inner] & (1u<<p.direction)) return 2;
                mask[p.inner]|=1u<<p.direction;
                if (px==x && py==y) ++centres;
            }
            if (centres!=4) return 3;
            for (int py=0;py<n;++py) for (int px=0;px<n;++px) {
                const int dx=px-x,dy=py-y;
                unsigned expected=0;
                if (std::max(std::abs(dx),std::abs(dy))<=5)
                    expected=(dy==0) | ((dx==0)<<1) | ((dx==-dy)<<2) | ((dx==dy)<<3);
                if (mask[py*n+px]!=expected) return 4;
                if (expected) ++changed;
            }
            unsigned start=0;
            for (unsigned g=0;g<9;++g) {
                for (;start<walk.groupEnds[g];++start) {
                    const int raw=walk.targets[start],tx=raw%(n+2)-1,ty=raw/(n+2)-1;
                    if (tx<0 || tx>=n || ty<0 || ty>=n || group[ty]*3+group[tx]!=int(g)) return 5;
                    if (affected[ty*n+tx]++) return 6;
                }
            }
            for (int ty=0;ty<n;++ty) for (int tx=0;tx<n;++tx) {
                bool reaches=false;
                for (int sy=std::max(0,ty-1);sy<=std::min(n-1,ty+1);++sy)
                    for (int sx=std::max(0,tx-1);sx<=std::min(n-1,tx+1);++sx)
                        reaches|=mask[sy*n+sx]!=0;
                if (bool(affected[ty*n+tx])!=reaches) return 7;
                ++destinations;
            }
            ++moves;
        }
    }
    std::cout << "{\"board_sizes\":18,\"move_origins\":" << moves << ",\"destination_checks\":"
              << destinations << ",\"changed_inner_cells\":" << changed << ",\"mismatches\":0}\n";
}

