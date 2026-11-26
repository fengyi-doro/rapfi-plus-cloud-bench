// SPDX-License-Identifier: GPL-3.0-or-later
#include "search/ab/pairing/search_hook.h"
#include <chrono>
#include <iostream>
#include <string>

using namespace f10;
using Clock = std::chrono::steady_clock;

static void emit(const Proof &p)
{
    std::cout << '[' << p.pre << ',' << p.mirror << ",[";
    for (size_t i = 0; i < p.pairs.size(); ++i) {
        auto [a,b] = p.pairs[i];
        if (i) std::cout << ',';
        std::cout << '[' << a << ',' << b << ']';
    }
    std::cout << "],[]]";
}

static void outputOrdinary(const f8::Witness &w)
{
    Proof p;
    if (w.preplay != f8::None) p.pre = w.preplay;
    for (int v = 0; v < Cells; ++v)
        if (w.mate[v] != f8::None && v < w.mate[v]) p.pairs.emplace_back(v, w.mate[v]);
    emit(p);
}

static void inspect(const std::string &name, const f10::Board &board)
{
    using PairingSearch::Cut;
    int ply = board.ply, first = ply % 2;
    const auto lower = PairingSearch::classify(board, Value(-1), VALUE_DRAW);
    const auto upper = PairingSearch::classify(board, VALUE_DRAW, Value(1));
    const auto exact = PairingSearch::classify(board, Value(-1), Value(1));
    if ((lower != Cut::None && lower != Cut::DrawLowerBound)
        || (upper != Cut::None && upper != Cut::DrawUpperBound)
        || (exact != Cut::None && exact != Cut::ExactDraw))
        throw std::logic_error("window polarity");
    std::cout << "[\"" << name << "\"," << ply << ',' << (lower != Cut::None) << ','
              << (upper != Cut::None) << ',' << (exact != Cut::None);
    f8::Bounds ordinary;
    f8::Witness lo, hi;
    if (!board.finished && ply < Cells && ply >= FirstSearchPly)
        ordinary = f8::prove(board, true, true, f8::MaxLive, false, &lo, &hi);
    std::cout << ',';
    if (ordinary.pairLo) outputOrdinary(lo); else std::cout << "null";
    std::cout << ',';
    if (ordinary.pairHi) outputOrdinary(hi); else std::cout << "null";
    bool eligible = !board.finished && ply < Cells && ply >= FirstSearchPly;
    auto start = Clock::now();
    auto black = eligible && board.rule == f8::Renju ? cheapBlack(board, !first) : CheapResult{};
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    std::cout << ',' << ns << ',';
    if (black.found) emit(black.proof); else std::cout << "null";
    Proof defending, attacking;
    bool draw = rootDraw(board, defending);
    std::cout << ',';
    if (draw) {
        if (!rootDefence(board, first, false, attacking)) throw std::logic_error("missing upper witness");
        emit(defending);
        std::cout << ',';
        emit(attacking);
    } else std::cout << "null,null";
    std::cout << "]\n";
}

int main() try {
    std::string name;
    int rule, count, includeLast;
    while (std::cin >> name >> rule >> count >> includeLast) {
        f10::Board board{f8::Rule(rule)};
        for (int i = 0, p; i < count; ++i) {
            inspect(name, board);
            if (!(std::cin >> p) || unsigned(p) >= Cells || !board.empty.has(p))
                throw std::logic_error("invalid corpus");
            board.put(p, i % 2);
        }
        if (includeLast) inspect(name, board);
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
