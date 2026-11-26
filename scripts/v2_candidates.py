"""Materialize the three measured hot-path trials separately against the frozen P1+P3 source."""
from pathlib import Path
from datasets import ROOT
from v2_build import TRIALS,trial

MIX="eval/mix9svqnnue.cpp"
BOARD="game/board.cpp"
BASE=TRIALS/"baseline"
CASES=TRIALS/"candidates"


def span(s,a,b,new):
    left=s.index(a)
    right=s.index(b,left)
    return s[:left]+new+s[right:]


def prepared_geometry(text):
    decl = r'''// Geometry is immutable for the board size. Each point still appears in the
// original directional order, including the four dependent updates at the centre.
struct UpdatePoint {
    uint16_t inner, outer;
    uint8_t direction, group;
    int power;
};
struct MoveGeometry {
    std::array<UpdatePoint,44> points {};
    std::array<uint16_t,169> targets {};
    std::array<uint8_t,9> groupEnds {};
    uint8_t pointCount = 0;
};
static const std::vector<MoveGeometry> &geometryFor(int size, const int8_t *groups)
{
    static std::array<std::once_flag,23> once;
    static std::array<std::vector<MoveGeometry>,23> bySize;
    std::call_once(once[size], [=] {
        auto &moves = bySize[size];
        moves.resize(size*size);
        for (int y=0; y<size; ++y)
            for (int x=0; x<size; ++x) {
                auto &move=moves[y*size+x];
                std::array<bool,22*22> touched {};
                for (int d=0; d<4; ++d)
                    for (int k=-5; k<=5; ++k) {
                        const int px=x+k*DX[d], py=y+k*DY[d];
                        if (px<0 || py<0 || px>=size || py>=size) continue;
                        move.points[move.pointCount++]={uint16_t(py*size+px),
                            uint16_t(py*(size+2)+px), uint8_t(d),
                            uint8_t(groups[py]*3+groups[px]), Power3[k+5]};
                        for (int ty=std::max(0,py-1); ty<=std::min(size-1,py+1); ++ty)
                            for (int tx=std::max(0,px-1); tx<=std::min(size-1,px+1); ++tx)
                                touched[ty*size+tx]=true;
                    }
                unsigned count=0;
                for (int g=0; g<9; ++g) {
                    for (int ty=0; ty<size; ++ty)
                        for (int tx=0; tx<size; ++tx)
                            if (touched[ty*size+tx] && groups[ty]*3+groups[tx]==g)
                                move.targets[count++]=uint16_t((ty+1)*(size+2)+tx+1);
                    move.groupEnds[g]=uint8_t(count);
                }
            }
    });
    return bySize[size];
}

'''
    text=text.replace('namespace Evaluation::mix9svq {','#include <mutex>\n\nnamespace Evaluation::mix9svq {',1)
    text=text.replace('class Accumulator :',decl+'class Accumulator :',1)
    text=text.replace(': MixAccumulatorState(boardSize) {}',': MixAccumulatorState(boardSize), geometry(geometryFor(boardSize, groupIndex)) {}',1)
    text=text.replace('    int getBucketIndex()', '    const std::vector<MoveGeometry> &geometry;\n    int getBucketIndex()',1)
    a=text.index('void Accumulator::move(')
    before,body=text[:a],text[a:]
    body=span(body,'    // Subtract value feature sum','    struct OnePointChange',r'''    const MoveGeometry &walk = geometry[y * boardSize + x];
    int newMapConvIdx = changeNum.outer;
    // Only real outputs reached by a 3x3 scatter receive a new version.
    // Other real cells retain the old row via the copied version table.
    for (int n=0; n<walk.groupEnds.back(); ++n) {
        versionOuterIndexTable[outerVersionIdxBase + walk.targets[n]] = newMapConvIdx;
        for (int b=0; b<ConvB::NumBatch; ++b)
            I16LS::store(mapConv[newMapConvIdx].data()+b*ConvB::RegWidth,I16Op::setzero());
        ++newMapConvIdx;
    }

''')
    body=body.replace('        int8_t   x;\n        int8_t   y;', '        uint16_t outer;\n        uint8_t  group;',1)
    body=span(body,'    int changeCount = 0;','    // Init value sum accumulator',r'''    const int changeCount = walk.pointCount;
    const int stone = pieceColor + 1;
    int newMapIdx = changeNum.inner;
    for (int i=0; i<changeCount; ++i) {
        const UpdatePoint &point = walk.points[i];
        const int dir = point.direction;
        OnePointChange &c = changeTable[i];
        c.outer = point.outer;
        c.group = point.group;
        c.mappingIdx = dir/2;
        auto &version = versionInnerIndexTable[innerVersionIdxBase+point.inner];
        c.oldMapIdx = version;
        c.newMapIdx = newMapIdx;
        const uint32_t oldShape = indexTable[c.oldMapIdx][dir];
        indexTable[newMapIdx] = indexTable[c.oldMapIdx];
        const uint32_t newShape = indexTable[newMapIdx][dir] = oldShape + stone*point.power;
        assert(newShape < ShapeNum);
        c.oldCodebookIdx = w.mapping_index[c.mappingIdx][oldShape];
        c.newCodebookIdx = w.mapping_index[c.mappingIdx][newShape];
        version = newMapIdx++;
        if (c.oldCodebookIdx != c.newCodebookIdx) {
            multiPrefetch<sizeof(int16_t)*FeatureDim>(w.codebook[c.mappingIdx][c.oldCodebookIdx]);
            multiPrefetch<sizeof(int16_t)*FeatureDim>(w.codebook[c.mappingIdx][c.newCodebookIdx]);
        }
    }

''')
    body=body.replace('outerIdxBase = c.y * outerBoardSize + c.x','outerIdxBase = c.outer',1)
    body=body.replace('vSumGroup[groupIndex[c.y]][groupIndex[c.x]]','vSumGroup[c.group / 3][c.group % 3]',1)
    body=span(body,'    // Add value feature sum','    // Move to next version',r'''    // Group the real destination rows so each group accumulator stays in registers.
    int point=0;
    for (int g=0; g<9; ++g) {
        auto &group = vSumGroup[g/3][g%3];
        I32Op::R sum[2*ConvB::NumBatch];
        for (int j=0; j<2*ConvB::NumBatch; ++j) sum[j]=group[j];
        for (; point<walk.groupEnds[g]; ++point) {
            const int oldIndex=versionOuterIndexTable[outerVersionIdxBasePrev+walk.targets[point]];
            auto *destination=mapConv[changeNum.outer+point].data();
            for (int b=0; b<ConvB::NumBatch; ++b) {
                auto oldF=I16LS::load(mapConv[oldIndex].data()+b*ConvB::RegWidth);
                auto delta=I16LS::load(destination+b*ConvB::RegWidth);
                auto newF=I16Op::add(oldF,delta);
                I16LS::store(destination+b*ConvB::RegWidth,newF);
                auto diff=I16Op::sub(I16Op::max(newF,I16Op::setzero()),
                                     I16Op::max(oldF,I16Op::setzero()));
                auto [lo,hi]=Convert<int16_t,int32_t>::convert(diff);
                sum[2*b]=I32Op::add(sum[2*b],lo);
                sum[2*b+1]=I32Op::add(sum[2*b+1],hi);
            }
        }
        for (int j=0; j<2*ConvB::NumBatch; ++j) group[j]=sum[j];
    }

''')
    return before+body


def packed_board(text):
    marker='            p2x[dir] = newP2x;'
    text=text.replace(marker,r'''#if defined(__BMI2__) && (defined(__i386__) || defined(__x86_64__))
            // Read before the one-byte write, avoiding a partial-store dependency.
            uint32_t previousDirections;
            std::memcpy(&previousDirections,p2x,sizeof(previousDirections));
#endif
'''+marker,1)
    first='            const int d1 = (dir + 1) & 3, d2 = (dir + 2) & 3, d3 = (dir + 3) & 3;'
    end='            PatternConfig::PCodeP4 cpBlack'
    start=text.index(first)
    old=text[start:text.index(end,start)]
    text=span(text,first,end,r'''#if defined(__BMI2__) && (defined(__i386__) || defined(__x86_64__))
            static_assert(PATTERN_NB==16 && sizeof(Pattern2x)==1);
            const uint32_t rotated=std::rotr(previousDirections,8*((dir+1)&3));
            const unsigned black=_pext_u32(rotated,0x000f0f0fu);
            const unsigned white=_pext_u32(rotated,0x00f0f0f0u);
            // The three unchanged patterns are exchangeable in either PCode table.
            const auto *rowBlack=PatternConfig::pcodeTable<R,BLACK>()[black>>8][(black>>4)&15][black&15];
            const auto *rowWhite=PatternConfig::pcodeTable<R,WHITE>()[white>>8][(white>>4)&15][white&15];
#else
'''+old+'#endif\n\n')
    return '#if defined(__BMI2__) && (defined(__i386__) || defined(__x86_64__))\n#include <immintrin.h>\n#endif\n'+text


def orbit(text):
    text=text.replace('namespace Evaluation::mix9svq {','#include "mix9_orbit.h"\n\nnamespace Evaluation::mix9svq {',1)
    return text.replace('simd::linearBatch4<','simd::linearOrbit4<')

def stable_pattern_counts(text):
    first="            st.p4Count[BLACK][p4Pair & 0xF]--;"
    last="            st.setLastPattern4(BLACK, p4Black, posi);"
    before=text.index(first,text.index("void Board::move(Pos pos)"))
    after=text.index(last,before)
    return text[:before]+'''            const uint8_t nextPair = packP4Pair(p4Black,p4White);
            // A different directional pattern can still have identical aggregate
            // classes for both colors. In that case both count deltas cancel.
            if (nextPair != p4Pair) {
                st.p4Count[BLACK][p4Pair & 0xF]--;
                st.p4Count[WHITE][p4Pair >> 4]--;
                st.p4Count[BLACK][p4Black]++;
                st.p4Count[WHITE][p4White]++;
                pattern4x2[posi] = nextPair;
            }

'''+text[after:]


def source(name,logical,transform):
    folder=CASES/name
    path=folder/logical
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(transform((BASE/logical).read_text(encoding="utf8")),encoding="utf8",newline="\n")
    return path


def generate():
    return {"geometry":{MIX:source("geometry",MIX,prepared_geometry)},
            "orbit":{MIX:source("orbit",MIX,orbit)},
            "pattern":{BOARD:source("pattern",BOARD,packed_board)},
            "counts":{BOARD:source("counts",BOARD,stable_pattern_counts)}}


if __name__=="__main__":
    import sys
    for name,files in generate().items():
        if len(sys.argv)==1 or name in sys.argv[1:]:
            trial(name,files)

