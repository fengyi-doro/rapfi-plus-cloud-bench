// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src_plus/search/ab/f15/certificate.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace f15;

static void emit(const std::string &game, int ply, int ordinary, const VcfResult &proof)
{
    std::cout << game << ' ' << ply << ' ' << ordinary;
    for (const Step &s : proof.line)
        std::cout << ' ' << s.gain << ',' << s.block << ',' << s.other << ',' << int(s.kind) << ',' << int(s.status);
    std::cout << '\n';
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "chosen") {
        int side, chosen, n;
        while (std::cin >> side >> chosen >> n) {
            Board position;
            for (int i = 0, p; i < n; ++i) {
                std::cin >> p;
                position = position.with(p, i % 2);
            }
            auto proof = Vcf(true).solve(position, side, chosen);
            emit("chosen", n, 0, proof);
        }
        return 0;
    }
    std::ifstream games("data/f23/renju_games.txt"), wins("data/f23/renju_wins.txt");
    std::map<std::pair<std::string, int>, int> ledger;
    std::string group, game;
    int ply, ordinary;
    while (wins >> game >> ply >> ordinary) ledger.emplace(std::make_pair(game, ply), ordinary);
    if (!games || ledger.size() != 252) return 2;
    int visited = 0, ordinaryWins = 0, augmentedWins = 0, addedWhite = 0;
    int rule, start, n;
    while (games >> group >> game >> rule >> start >> n) {
        Board position;
        for (int i = 0; i <= n; ++i) {
            if (group == "main" && rule == 2 && i >= start && (start == n || i < n)) {
                ++visited;
                auto base = Vcf(false).solve(position, i % 2);
                auto result = Vcf(true).solve(position, i % 2);
                auto expected = ledger.find({game, i});
                if (result.exhausted || result.win != (expected != ledger.end())
                    || base.win != (expected != ledger.end() && expected->second == 1)) return 3;
                if (result.win) {
                    if (!replayCertificate(position, i % 2, result.line)) return 4;
                    emit(game, i, base.win, result);
                    ++augmentedWins;
                    ordinaryWins += base.win;
                    if (i % 2 && !base.win) {
                        ++addedWhite;
                        auto capped = Vcf(true, {1024}).solve(position, 1);
                        if (!capped.win || !replayCertificate(position, 1, capped.line)) return 5;
                    }
                }
            }
            if (i < n) {
                int p;
                if (!(games >> p)) return 6;
                position = position.with(p, i % 2);
            }
        }
    }
    return visited == 9069 && ordinaryWins == 172 && augmentedWins == 252 && addedWhite == 41 ? 0 : 7;
}
