// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "simd/linear.h"

#if defined(__clang__) && defined(__x86_64__) && defined(USE_AVX2) && !defined(USE_AVX512) && !defined(RAPFI_AVX2_ONLY)
#define RAPFI_ORBIT_VNNI_DISPATCH
#include <immintrin.h>
#include <cpuid.h>
#endif

namespace Evaluation::simd {
#ifdef RAPFI_ORBIT_VNNI_DISPATCH
inline bool hasOrbitDotProduct()
{
    static const bool present = [] {
        unsigned a,b,c,d;
        if (!__get_cpuid_count(7,0,&a,&b,&c,&d) || a<1) return false;
        __cpuid_count(7,1,a,b,c,d);
        return (a & bit_AVXVNNI) && (d & bit_AVXVNNIINT8);
    }();
    return present;
}

// Two adjacent 8-output vectors reuse the same four packed input words.
// The nonsaturating signed or unsigned VNNI dot adds modulo 2^32.
template <int Rows, int Cols, bool Signed>
__attribute__((target("avxvnni,avxvnniint8"),noinline))
void orbitDot4(int32_t (&output)[4][Rows],const std::array<const int8_t *,4> &input,
               const int8_t *weight,const int32_t *bias)
{
    static_assert(Rows%16==0 && Cols%4==0);
    for (int row=0; row<Rows; row+=16) {
        __m256i sum[4][2];
        for (int k=0;k<4;++k) {
            sum[k][0]=_mm256_load_si256(reinterpret_cast<const __m256i *>(bias+row));
            sum[k][1]=_mm256_load_si256(reinterpret_cast<const __m256i *>(bias+row+8));
        }
        for (int col=0;col<Cols;col+=4) {
            const auto *pair=reinterpret_cast<const __m256i *>(weight+col*Rows+4*row);
            const auto lo=_mm256_load_si256(pair),hi=_mm256_load_si256(pair+1);
            for (int k=0;k<4;++k) {
                int32_t word;
                std::memcpy(&word,input[k]+col,sizeof(word));
                const auto v=_mm256_set1_epi32(word);
                if constexpr (Signed) {
                    sum[k][0]=_mm256_dpbssd_epi32(sum[k][0],v,lo);
                    sum[k][1]=_mm256_dpbssd_epi32(sum[k][1],v,hi);
                } else {
                    sum[k][0]=_mm256_dpbusd_avx_epi32(sum[k][0],v,lo);
                    sum[k][1]=_mm256_dpbusd_avx_epi32(sum[k][1],v,hi);
                }
            }
        }
        for (int k=0;k<4;++k) {
            _mm256_store_si256(reinterpret_cast<__m256i *>(output[k]+row),sum[k][0]);
            _mm256_store_si256(reinterpret_cast<__m256i *>(output[k]+row+8),sum[k][1]);
        }
    }
}
#endif

template<int Rows,int Cols,bool Signed=false>
inline void linearOrbit4(int32_t (&output)[4][Rows],const std::array<const int8_t *,4> &input,
                         const int8_t *weight,const int32_t *bias)
{
#ifdef RAPFI_ORBIT_VNNI_DISPATCH
    if (hasOrbitDotProduct()) {
        orbitDot4<Rows,Cols,Signed>(output,input,weight,bias);
        return;
    }
#endif
    linearBatch4<Rows,Cols,Signed>(output,input,weight,bias);
}
} // namespace Evaluation::simd

