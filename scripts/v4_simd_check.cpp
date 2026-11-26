// SPDX-License-Identifier: GPL-3.0-or-later
#include "eval/simdops.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>

using namespace Evaluation;
static std::mt19937 rng(358);

// Use unsigned scalar accumulation so extreme int32 biases also test wraparound.
template<int Rows, int Cols, typename T, bool Signed, bool PreRelu = false>
static int affine() {
    alignas(64) T raw[Rows*Cols], packed[Rows*Cols], input[Cols];
    alignas(64) int32_t bias[Rows], actual[Rows];
    constexpr int trials = 128;
    for (int trial=0; trial<trials; ++trial) {
        for (auto &w : raw)
            w = trial<2 ? (trial==0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max())
                        : std::bit_cast<T>(std::make_unsigned_t<T>(rng()));
        std::copy(std::begin(raw),std::end(raw),packed);
        simd::preprocessLinear<Rows,Cols>(packed);
        for (auto &x : input) {
            x = trial<2 ? (trial==0 && (Signed || PreRelu) ? std::numeric_limits<T>::min()
                                                            : std::numeric_limits<T>::max())
                        : std::bit_cast<T>(std::make_unsigned_t<T>(rng()));
            if constexpr (!Signed && !PreRelu) x = T(uint8_t(x)&127);
        }
        for (auto &b : bias) b = trial==0 ? INT32_MIN : trial==1 ? INT32_MAX
                                                               : std::bit_cast<int32_t>(uint32_t(rng()));
        simd::linear<Rows,Cols,Signed,true,PreRelu>(actual,input,packed,bias);
        for (int r=0; r<Rows; ++r) {
            uint32_t expected=uint32_t(bias[r]);
            for (int c=0; c<Cols; ++c) {
                int x=PreRelu ? std::max(0,int(input[c])) : int(input[c]);
                expected+=uint32_t(x*int(raw[r*Cols+c]));
            }
            if (uint32_t(actual[r])!=expected) {
                std::cerr << Rows << 'x' << Cols << " bits=" << sizeof(T)*8
                          << " signed=" << Signed << " pre_relu=" << PreRelu << " trial=" << trial
                          << " row=" << r << " actual=" << uint32_t(actual[r]) << " expected=" << expected
                          << " bias=" << bias[r] << " x0=" << int(input[0]) << " w0=" << int(raw[r*Cols]) << '\n';
                throw std::logic_error("scalar affine mismatch");
            }
        }
    }
    return trials*Rows;
}

static void bench64() {
    alignas(64) int8_t weights[64*64], inputs[128][64];
    alignas(64) int32_t bias[64], output[64];
    for (auto &w : weights) w = std::bit_cast<int8_t>(uint8_t(rng()));
    for (auto &row : inputs) for (auto &v : row) v = int8_t(rng() & 127);
    for (auto &v : bias) v = int32_t(rng() % 10001)-5000;
    simd::preprocessLinear<64,64>(weights);
    constexpr int calls = 1 << 17;
    for (int block=0; block<5; ++block) {
        auto start=std::chrono::steady_clock::now();
        for (int i=0; i<calls; ++i) {
            simd::linear<64,64>(output,inputs[i%128],weights,bias);
            asm volatile("" : : "m"(output) : "memory");
        }
        auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start);
        if (block) std::cout << double(elapsed.count())/calls << '\n';
    }
}

int main(int argc,char **argv) {
    if (argc==2 && std::string_view(argv[1])=="--bench") { bench64(); return 0; }
    int words=affine<64,64,int8_t,false>() + affine<64,320,int8_t,false>()
              + affine<4,64,int8_t,false>() + affine<128,64,int8_t,false>()
              + affine<64,64,int8_t,true>() + affine<528,64,int8_t,false>()
              + affine<16,32,int16_t,true>() + affine<16,32,int16_t,false,true>();
    std::cout << "{\"scalar_i32_words\":" << words << ",\"mismatches\":0}\n";
}
