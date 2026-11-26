// SPDX-License-Identifier: GPL-3.0-or-later
#include "eval/simdops.h"
#if __has_include("mix9_orbit.h")
#include "mix9_orbit.h"
#define TEST_V2_ORBIT
#elif __has_include("eval/mix9_orbit.h")
#include "eval/mix9_orbit.h"
#define TEST_V2_ORBIT
#endif
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

template <int Rows, bool Signed>
bool affine(std::mt19937 &rng)
{
    constexpr int Cols = 64;
    alignas(32) int8_t weights[Rows * Cols], input[4][Cols];
    alignas(32) int32_t biases[Rows], old[4][Rows], batch[4][Rows];
    for (int t = 0; t < 1024; ++t) {
        for (auto &w : weights)
            w = t == 0 ? -128 : t == 1 ? 127 : std::bit_cast<int8_t>(uint8_t(rng()));
        for (auto &b : biases)
            b = t % 3 == 0 ? INT32_MAX : t % 3 == 1 ? INT32_MIN : std::bit_cast<int32_t>(uint32_t(rng()));
        for (int k = 0; k < 4; ++k) {
            for (auto &v : input[k]) {
                if constexpr (Signed)
                    v = t == 0 ? -128 : std::bit_cast<int8_t>(uint8_t(rng()));
                else
                    v = t == 0 ? 127 : int8_t(rng() & 127);
            }
            Evaluation::simd::linear<Rows, Cols, Signed>(old[k], input[k], weights, biases);
        }
        Evaluation::simd::linearBatch4<Rows, Cols, Signed>(batch, {input[0],input[1],input[2],input[3]},
                                                           weights, biases);
        if (std::memcmp(old, batch, sizeof(old))) return false;
#ifdef TEST_V2_ORBIT
        Evaluation::simd::linearOrbit4<Rows,Cols,Signed>(batch,{input[0],input[1],input[2],input[3]},weights,biases);
        if (std::memcmp(old,batch,sizeof(old))) return false;
#endif
    }
    return true;
}

int main()
{
    std::mt19937 rng(358);
    if (!affine<128,false>(rng) || !affine<64,true>(rng) || !affine<64,false>(rng))
        return 1;
#ifdef TEST_V2_ORBIT
    std::cout << "{\"trials_per_shape\":1024,\"int32_words\":2097152,\"mismatches\":0";
#if defined(__clang__) && defined(__x86_64__) && defined(USE_AVX2) && !defined(USE_AVX512)
    std::cout << ",\"native_vnni_int8\":" << (Evaluation::simd::hasOrbitDotProduct()?"true":"false");
#endif
    std::cout << "}\n";
#else
    std::cout << "{\"trials_per_shape\":1024,\"int32_words\":1048576,\"mismatches\":0}\n";
#endif
}
