#include "rootprior.h"

#include "../../core/iohelper.h"
#include "../../eval/evaluator.h"
#include "../../eval/scoretables.h"
#include "../../game/board.h"
#include "../searchthread.h"
#include "rzone.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <stdexcept>

#ifdef __linux__
#include <cerrno>
#include <sys/socket.h>
#endif

namespace Search::AB {
namespace {

using Policy = Evaluation::PolicyBuffer;

// Replay through the normal evaluator hooks. The scratch board has no attached evaluator.
void replayOrientation(const Board &source, Board &board, Evaluation::Evaluator &ev,
                       TransformType transform)
{
    board.newGame(ev.rule);
    ev.initEmptyBoard();
    for (int i = 0; i < source.ply(); ++i) {
        const Pos move = source.getHistoryMove(i);
        if (move == Pos::PASS) {
            board.move(ev.rule, move);
            ev.afterPass(board);
        } else {
            const Pos p = applyTransform(move, source.size(), transform);
            ev.beforeMove(board, p);
            board.move(ev.rule, p);
            ev.afterMove(board, p);
        }
    }
}

void evaluateD4(SearchThread &thread, RootPrior &prior)
{
    const Board &root = *thread.board;
    auto &ev = *thread.evaluator;
    Board rotated(root.size());
    std::array<int, FULL_BOARD_CELL_COUNT> sums {};
    float winRate = 0;
    // Finish in the original orientation so the ordinary recursive search inherits
    // a synchronized accumulator, including the side that did not just evaluate.
    for (int i = 1; i <= TRANS_NB; ++i) {
        const auto transform = TransformType(i % TRANS_NB);
        replayOrientation(root, rotated, ev, transform);
        Policy policy(root.size());
        for (const auto &move : thread.rootMoves)
            policy.setComputeFlag(applyTransform(move.pv[0], root.size(), transform));
        ev.evaluatePolicy(rotated, policy);
        winRate += ev.evaluateValue(rotated).winningRate();
        for (const auto &move : thread.rootMoves) {
            const Pos p = move.pv[0];
            sums[p] += policy.score(applyTransform(p, root.size(), transform));
        }
    }
    for (const auto &move : thread.rootMoves)
        prior.policy[move.pv[0]] = Score(std::lround(double(sums[move.pv[0]]) / TRANS_NB));
    prior.value = Evaluation::winRateToValue(winRate / TRANS_NB);
}

#ifdef __linux__
void transfer(int fd, void *buffer, size_t length, bool sending)
{
    auto data = static_cast<unsigned char *>(buffer);
    while (length) {
        ssize_t n = sending ? send(fd, data, length, MSG_NOSIGNAL) : recv(fd, data, length, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("KataGomo CPU oracle disconnected");
        data += n;
        length -= size_t(n);
    }
}

void evaluateKata(SearchThread &thread, RootPrior &prior)
{
    const char *socketText = std::getenv("RAPFI_ORACLE_FD");
    if (!socketText)
        throw std::runtime_error("KataGomo CPU oracle socket is missing");
    const int fd = std::stoi(socketText);
    const Board &board = *thread.board;
    const Color side = board.sideToMove();
    const Pos last = board.getLastMove();
    auto &ev = *thread.evaluator;
    Policy local(15);
    for (const auto &move : thread.rootMoves)
        local.setComputeFlag(move.pv[0]);
    ev.evaluatePolicy(board, local);
    float localWin = ev.evaluateValue(board).winningRate();
    Score bestLocal = std::numeric_limits<Score>::min(), second = bestLocal;
    for (const auto &move : thread.rootMoves) {
        const Score s = local.score(move.pv[0]);
        if (s >= bestLocal) {
            second = bestLocal;
            bestLocal = s;
        } else
            second = std::max(second, s);
    }
    if (prior.orderingOnly && (localWin < .2f || localWin > .8f || bestLocal - second > 1.5f * Policy::ScoreScale))
        return;
    std::array<uint8_t, 228> request {};
    request[0] = uint8_t(thread.options().rule.rule);
    request[1] = uint8_t(side);
    request[2] = board.isInBoard(last) ? uint8_t(last.y() * 15 + last.x()) : 255;
    for (int i = 0; i < 225; ++i) {
        Color piece = board.get(Pos(i % 15, i / 15));
        request[3 + i] = piece == side ? 1 : piece == ~side ? 2 : 0;
    }
    const char *usage = std::getenv("RAPFI_ORACLE_USAGE");
    if (usage && std::string_view(usage) == "children") {
        std::vector<Pos> moves;
        for (const auto &move : thread.rootMoves) {
            moves.push_back(move.pv[0]);
            prior.policy[move.pv[0]] = local.score(move.pv[0]);
        }
        const size_t count = std::min<size_t>(6, moves.size());
        std::partial_sort(moves.begin(), moves.begin() + count, moves.end(),
                          [&](Pos a, Pos b) { return local.score(a) > local.score(b); });
        std::array<uint8_t, 7> candidates {};
        candidates[0] = uint8_t(count);
        for (size_t j = 0; j < count; ++j)
            candidates[j+1] = uint8_t(moves[j].y()*15 + moves[j].x());
        request[0] |= 128;
        transfer(fd, request.data(), request.size(), true);
        transfer(fd, candidates.data(), count+1, true);
        std::array<float, 6> scores {};
        transfer(fd, scores.data(), count*sizeof(float), false);
        float mean = 0, best = 0;
        for (size_t j = 0; j < count; ++j) {
            if (!std::isfinite(scores[j]) || scores[j] < 0 || scores[j] > 1)
                throw std::runtime_error("invalid root-child WDL");
            mean += scores[j] / float(count);
            best = std::max(best, scores[j]);
        }
        // Centre the child scores within this root. The teacher and local
        // head can have different WDL calibration even on the same board.
        for (size_t j = 0; j < count; ++j) {
            float logits = std::clamp(4.f * (scores[j] - mean), -.9f, .9f);
            prior.policy[moves[j]] = Score(std::clamp(int(local.score(moves[j])) + int(std::lround(logits*Policy::ScoreScale)),
                                                       -32000, 32000));
        }
        prior.value = Evaluation::winRateToValue(localWin + .20f*std::clamp(best-localWin, -.25f, .25f));
        prior.childCount = unsigned(count);
        return;
    }
    std::array<float, 226> response;
    transfer(fd, request.data(), request.size(), true);
    transfer(fd, response.data(), sizeof(response), false);
    const float globalWinRate = response.back();
    if (!std::isfinite(globalWinRate) || globalWinRate < 0 || globalWinRate > 1)
        throw std::runtime_error("invalid KataGomo WDL");
    float bestGlobal = -std::numeric_limits<float>::infinity();
    for (const auto &move : thread.rootMoves) {
        const Pos p = move.pv[0];
        const float logit = response[p.y() * 15 + p.x()];
        if (!std::isfinite(logit))
            throw std::runtime_error("invalid KataGomo policy");
        bestGlobal = std::max(bestGlobal, logit);
    }
    for (const auto &move : thread.rootMoves) {
        const Pos p = move.pv[0];
        float global = bestLocal + Policy::ScoreScale * (response[p.y() * 15 + p.x()] - bestGlobal);
        float score = .4f * local.score(p) + .6f * global;
        if (prior.orderingOnly) {
            const float correction = std::clamp(.35f * (global - local.score(p)),
                                                -1.5f * Policy::ScoreScale, 1.5f * Policy::ScoreScale);
            score = local.score(p) + correction;
        }
        prior.policy[p] = Score(std::lround(std::clamp(score, -32000.f, 32000.f)));
    }
    const float globalWeight = prior.orderingOnly ? .25f : .5f;
    prior.value = Evaluation::winRateToValue(localWin + globalWeight * (globalWinRate - localWin));
}
#endif

}  // namespace

void prepareRootPrior(SearchThread &thread, RootPrior &prior, std::string_view experiment)
{
    prior.value = VALUE_NONE;
    prior.orderingOnly = false;
    prior.childCount = 0;
    if (experiment != "d4" && experiment != "kata")
        return;
    const auto &board = *thread.board;
    const auto &opts = thread.options();
    const char *usage = std::getenv("RAPFI_ORACLE_USAGE");
    prior.orderingOnly = experiment == "kata" && usage
                         && (std::string_view(usage) == "selective" || std::string_view(usage) == "children");
    if (prior.orderingOnly && thread.engine.ctx.timectl.optimum() < 45)
        return;
    if (!thread.evaluator || !opts.timeLimit || opts.balanceMode || board.size() != 15
        || board.ply() < 10 || board.ply() > 80 || opts.timeLeft < 400 || thread.rootMoves.size() < 4)
        return;
    for (Color side : {BLACK, WHITE})
        if (board.p4Count(side, A_FIVE) || board.p4Count(side, B_FLEX4)
            || board.p4Count(side, C_BLOCK4_FLEX3))
            return;
    auto start = std::chrono::steady_clock::now();
    prior.policy.fill(std::numeric_limits<Score>::min());
    if (experiment == "d4")
        evaluateD4(thread, prior);
    else {
#ifdef __linux__
        evaluateKata(thread, prior);
#else
        throw std::runtime_error("KataGomo CPU oracle requires a Linux socket");
#endif
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (prior.childCount)
        MESSAGEL("ROOT_PRIOR " << experiment << ' ' << elapsed << ' ' << prior.childCount);
    else if (prior.value != VALUE_NONE)
        MESSAGEL("ROOT_PRIOR " << experiment << ' ' << elapsed);
    else
        MESSAGEL("ROOT_GATE_NS " << elapsed);
}

Pos prepareRootProof(SearchThread &thread, Value &mate)
{
    using namespace RZone;
    const Board &root = *thread.board;
    const auto &opts = thread.options();
    if (!opts.timeLimit || opts.balanceMode || opts.multiPV != 1 || root.size() != Width
        || root.ply() < 8 || opts.timeLeft < 100)
        return Pos::NONE;
    // A continuous-four attack must start from a four-producing move. The native
    // board can cheaply reject roots without one, before building the proof position.
    bool forcing = false;
    for (Color side : {BLACK, WHITE})
        for (int p = E_BLOCK4; p <= A_FIVE; ++p)
            forcing |= root.p4Count(side, Pattern4(p)) != 0;
    if (!forcing)
        return Pos::NONE;
    auto start = std::chrono::steady_clock::now();
    Position position(root, opts.rule);
    const Color us = root.sideToMove();
    Work attack {0, 240}, defence {0, 160};
    Proof ourLine = findWin(position, us, 15, attack);
    Pos winning = Pos::NONE;
    size_t removed = 0;
    if (!ourLine.empty() && verify(position, us, ourLine)) {
        const int cell = ourLine.front().attack;
        const Pos p(cell % Width, cell / Width);
        if (std::find(thread.rootMoves.begin(), thread.rootMoves.end(), p) != thread.rootMoves.end()) {
            winning = p;
            int plies = int(2 * ourLine.size()) + (ourLine.back().kind == End::Five ? -1 : 1);
            mate = mate_in(plies);
        }
    }
    if (!winning) {
        // Pass is only a hypothesis. Exclude a real reply solely after replaying
        // the opponent's entire certificate on the actual child position.
        const Proof opponent = findWin(position, ~us, 13, defence);
        if (!opponent.empty()) {
            Mask losing;
            for (const auto &move : thread.rootMoves) {
                const int cell = cellId(move.pv[0]);
                position.put(cell, us);
                if (!position.winsOn(cell, us) && verify(position, ~us, opponent))
                    losing.add(cell);
                position.take(cell);
            }
            if (losing.any() && size_t(losing.size()) < thread.rootMoves.size()) {
                removed = losing.size();
                for (auto &worker : thread.engine)
                    std::erase_if(worker->rootMoves, [&](const RootMove &move) {
                        return losing.has(cellId(move.pv[0]));
                    });
            }
        }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    MESSAGEL("ROOT_PROOF " << int(bool(winning)) << ' ' << removed << ' ' << attack.nodes + defence.nodes << ' ' << elapsed);
    return winning;
}

namespace {

using ProofClock = std::chrono::steady_clock;

// The first threat need not give check. A VCF continuation must work after
// every legal defender reply, including remote counterattacks and wins.
int proveAllReplies(RZone::Position &pos, Color us, RZone::Work &work,
                    ProofClock::time_point deadline, unsigned &covered)
{
    using namespace RZone;
    if (pos.targets(~us, 4).any())
        return 0;
    Proof primary = findWin(pos, us, 13, work);
    Mask critical;
    if (primary.empty() || !verify(pos, us, primary, &critical))
        return 0;

    Mask ordinary = pos.empties();
    for (size_t w = 0; w < ordinary.words.size(); ++w)
        ordinary.words[w] &= ~critical.words[w];
    std::vector<Proof> lines {std::move(primary)};
    int worst = 0;
    unsigned examined = 0;
    // Check the witness's area of influence first. Even outside that area,
    // replay the line on the actual child before accepting the root move.
    for (Mask replies : {critical, ordinary}) {
        for (int r; (r = replies.pop()) != NoCell;) {
            if (ProofClock::now() >= deadline)
                return 0;
            if (pos.forbidden(r, ~us))
                continue;
            pos.put(r, ~us);
            int length = 0;
            if (!pos.winsOn(r, ~us)) {
                auto line = std::find_if(lines.begin(), lines.end(), [&](const Proof &p) {
                    return verify(pos, us, p);
                });
                if (line == lines.end()) {
                    Proof alternative = findWin(pos, us, 13, work);
                    if (!alternative.empty() && verify(pos, us, alternative)) {
                        lines.push_back(std::move(alternative));
                        line = std::prev(lines.end());
                    }
                }
                if (line != lines.end())
                    length = int(2 * line->size()) + (line->back().kind == End::Five ? -1 : 1);
            }
            pos.take(r);
            if (!length)
                return 0;
            worst = std::max(worst, length);
            ++examined;
        }
    }
    covered = examined;
    return examined ? 2 + worst : 0;
}

}  // namespace

Pos prepareRootThreeProof(SearchThread &thread, Value &mate)
{
    using namespace RZone;
    const Board &root = *thread.board;
    const auto &opts = thread.options();
    const Color us = root.sideToMove();
    if (!opts.timeLimit || opts.balanceMode || opts.multiPV != 1 || !thread.evaluator || root.size() != Width
        || root.ply() < 10 || root.ply() > 84 || opts.timeLeft < 350 || root.p4Count(~us, A_FIVE)
        || (!root.p4Count(us, H_FLEX3) && !root.p4Count(us, G_FLEX3_PLUS) && !root.p4Count(us, F_FLEX3_2X)))
        return Pos::NONE;

    auto start = ProofClock::now();
    auto budget = std::clamp<Time>(thread.engine.ctx.timectl.optimum() / 22, 1, 3);
    auto deadline = start + std::chrono::milliseconds(budget);
    std::vector<Pos> candidates;
    Policy local(Width);
    for (const auto &move : thread.rootMoves) {
        const Pos p = move.pv[0];
        const Pattern4 pattern = root.pattern4(p, us);
        if (pattern >= H_FLEX3 && pattern <= F_FLEX3_2X) {
            candidates.push_back(p);
            local.setComputeFlag(p);
        }
    }
    Work work {0, 1100};
    unsigned attempts = 0, covered = 0;
    Pos winning = Pos::NONE;
    if (!candidates.empty()) {
        thread.evaluator->evaluatePolicy(root, local);
        std::stable_sort(candidates.begin(), candidates.end(), [&](Pos a, Pos b) {
            return local.score(a) > local.score(b);
        });
        Position position(root, opts.rule);
        for (Pos p : candidates) {
            if (attempts >= 4 || work.nodes >= work.limit || ProofClock::now() >= deadline)
                break;
            ++attempts;
            const int cell = cellId(p);
            if (position.forbidden(cell, us))
                continue;
            position.put(cell, us);
            int plies = proveAllReplies(position, us, work, deadline, covered);
            position.take(cell);
            if (plies) {
                winning = p;
                mate = mate_in(plies);
                break;
            }
        }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(ProofClock::now() - start).count();
    MESSAGEL("ROOT_THREE_PROOF " << attempts << ' ' << int(bool(winning)) << ' ' << covered << ' ' << work.nodes << ' ' << elapsed);
    return winning;
}

}  // namespace Search::AB
