// SPDX-License-Identifier: GPL-3.0-or-later
// Warm-kernel ablation of the 528-output policy weight generator at the release's packed layout.
#include "eval/simdops.h"
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>

using namespace Evaluation;
constexpr int Rows=528, Cols=64;
alignas(64) static int8_t weights[Rows*Cols], inputs[128][Cols];
alignas(64) static int32_t biases[Rows], output[Rows];

[[gnu::noinline]] static void released(const int8_t *input) {
    simd::linear<Rows,Cols>(output,input,weights,biases);
}

template<int N>
static void tile(int row, const int8_t *input) {
    __m256i sum[N];
    for (int j=0; j<N; ++j) sum[j]=_mm256_load_si256(reinterpret_cast<const __m256i*>(biases+row+8*j));
    for (int col=0; col<Cols; col+=4) {
        int32_t packed;
        std::memcpy(&packed,input+col,sizeof(packed));
        auto in=_mm256_set1_epi32(packed);
        const auto *w=reinterpret_cast<const __m256i*>(weights+col*Rows+4*row);
        for (int j=0; j<N; ++j) sum[j]=_mm256_dpbusd_avx_epi32(sum[j],in,_mm256_load_si256(w+j));
    }
    for (int j=0; j<N; ++j) _mm256_store_si256(reinterpret_cast<__m256i*>(output+row+8*j),sum[j]);
}
[[gnu::noinline]] static void tiled(const int8_t *input) {
    for (int row=0; row<512; row+=64) tile<8>(row,input);
    tile<2>(512,input);
}

int main() {
    std::mt19937 rng(358);
    for (auto &w:weights) w=std::bit_cast<int8_t>(uint8_t(rng()));
    for (auto &row:inputs) for (auto &v:row) v=rng()&127;
    for (auto &v:biases) v=std::bit_cast<int32_t>(uint32_t(rng()));
    simd::preprocessLinear<Rows,Cols>(weights);
    for (auto &input:inputs) {
        alignas(64) int32_t expected[Rows];
        released(input);
        std::memcpy(expected,output,sizeof(output));
        tiled(input);
        if (std::memcmp(output,expected,sizeof(output))) return 1;
    }
    std::cout << "rep,variant,ns_per_call\n";
    for (int repeat=0; repeat<16; ++repeat) for (int leg=0; leg<2; ++leg) {
        bool trial=(repeat+leg)%2;
        auto fn=trial?tiled:released;
        auto start=std::chrono::steady_clock::now();
        constexpr int calls=16384;
        for (int i=0; i<calls; ++i) {
            fn(inputs[i%128]);
            asm volatile("" : : "m"(output) : "memory");
        }
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        std::cout << repeat << ',' << (trial?"tile64":"integrated") << ',' << double(ns)/calls << '\n';
    }
}
