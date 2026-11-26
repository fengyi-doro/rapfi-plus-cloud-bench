// SPDX-License-Identifier: GPL-3.0-or-later
#include "search/ab/rzone.h"
#include <chrono>
#include <iostream>
#include <windows.h>

using namespace Search::RZone;
int main(int argc, char **argv) try
{
    if (argc == 2 && std::string_view(argv[1]) == "fixture") {
        Position board(RENJU);
        int n, extra, count;
        std::cin >> n >> extra >> count;
        for (int i = 0, p; i < n; ++i) {
            std::cin >> p;
            board.put(p, Color(i % 2));
        }
        Proof proof;
        for (int i = 0; i < count; ++i) {
            int a, b, other, kind;
            std::cin >> a >> b >> other >> kind;
            proof.push_back({a,b,other,End(kind)});
        }
        Mask zone;
        if (!verify(board,BLACK,proof,&zone) || !zone.has(extra)) return 4;
        board.put(extra,WHITE);
        if (!board.forbidden(proof.front().attack,BLACK) || verify(board,BLACK,proof)) return 5;
        std::cout << "recursive forbidden read in zone " << extra << '\n';
        return 0;
    }
    const bool audit = argc == 1;
    const int repeats = audit ? 1 : std::stoi(argv[1]);
    int id, ruleId, n;
    while (std::cin >> id >> ruleId >> n) {
        Position board{Rule(ruleId)};
        for (int i = 0, p; i < n; ++i) {
            std::cin >> p;
            board.put(p, Color(i % 2));
        }
        for (int rep = 0; rep < repeats; ++rep) {
            DefenseMap answers[2];
            for (int j = 0; j < 2; ++j) {
                int mode = (id + rep + j) % 2;
                ULONG64 before, after;
                if (!QueryThreadCycleTime(GetCurrentThread(), &before)) return 3;
                auto start = std::chrono::steady_clock::now();
                auto &a = answers[mode] = defend(board, Color(n % 2), 9, mode, audit && mode);
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
                if (!QueryThreadCycleTime(GetCurrentThread(), &after)) return 3;
                std::cout << "MAP " << id << ' ' << rep << ' ' << mode << ' ' << a.nodes << ' '
                          << a.skipped << ' ' << a.replays << ' ' << a.promoted << ' ' << ns.count()
                          << ' ' << after - before << ' ';
                for (int p = 0; p < Cells; ++p)
                    std::cout << (a.losses.has(p) ? 'L' : a.legal.has(p) ? '.' : 'X');
                std::cout << '\n';
            }
            if (answers[0].legal != answers[1].legal || answers[0].losses != answers[1].losses)
                throw std::logic_error("defense classification mismatch");
        }
    }
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
