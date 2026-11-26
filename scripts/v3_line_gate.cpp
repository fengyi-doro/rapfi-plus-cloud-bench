// SPDX-License-Identifier: GPL-3.0-or-later
#include "game/pattern.h"
#include <cstdio>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

template <Rule R>
void emit()
{
    constexpr unsigned bits = 2 * PatternConfig::HalfLineLen<R>;
    for (unsigned lo = 0; lo < (1u << bits); ++lo)
        for (unsigned hi = 0; hi < (1u << bits); ++hi) {
            uint64_t key = lo | (uint64_t(3) << bits) | (uint64_t(hi) << (bits + 2));
            if (!PatternConfig::isValidLineKey<R>(key)) continue;
            Pattern2x p = PatternConfig::lookupPattern<R>(key);
            const uint8_t triple[] = {uint8_t(p[BLACK] | (p[WHITE] << 4)),
                                      PatternConfig::lookupDefenceTable<R>(key, BLACK),
                                      PatternConfig::lookupDefenceTable<R>(key, WHITE)};
            std::fwrite(triple, 1, sizeof(triple), stdout);
        }
}

int main()
{
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    emit<FREESTYLE>();
    emit<STANDARD>();
    emit<RENJU>();
    return std::ferror(stdout) ? 1 : 0;
}

