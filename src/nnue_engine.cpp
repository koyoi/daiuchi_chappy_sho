#include "nnue_engine.h"

#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace shogi {

namespace {

constexpr int MateScore = 100000;
constexpr int ExactScore = 0;
constexpr int LowerBound = 1;
constexpr int UpperBound = 2;

int pieceValue(PieceType type) {
    switch (type) {
    case Pawn: return 100;
    case Lance: return 300;
    case Knight: return 320;
    case Silver: return 520;
    case Gold: return 620;
    case Bishop: return 850;
    case Rook: return 1050;
    case ProPawn: case ProLance: case ProKnight: case ProSilver: return 560;
    case Horse: return 1150;
    case Dragon: return 1350;
    default: return 0;
    }
}

void gravityUpdate(std::int16_t& entry, int depth, bool good) {
    const int bonus = good ? depth * depth : -(depth * depth);
    int val = static_cast<int>(entry) + bonus - static_cast<int>(entry) * std::abs(bonus) / 16384;
    entry = static_cast<std::int16_t>(std::clamp(val, -16384, 16384));
}

int detectPhysicalCores() {
#ifdef _WIN32
    DWORD length = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) && GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return 0;
    std::vector<unsigned char> buffer(length);
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length))
        return 0;
    int cores = 0;
    DWORD offset = 0;
    while (offset < length) {
        auto* current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (current->Relationship == RelationProcessorCore) ++cores;
        offset += current->Size;
    }
    return cores;
#else
    return 0;
#endif
}

int defaultThreadCount() {
    int cores = detectPhysicalCores();
    if (cores <= 0) {
        const unsigned int logical = std::thread::hardware_concurrency();
        cores = logical > 0 ? static_cast<int>((logical + 1) / 2) : 1;
    }
    return std::max(1, cores - 2);
}

int chebyshevDistance(int left, int right) {
    return std::max(std::abs(fileOf(left) - fileOf(right)), std::abs(rankOf(left) - rankOf(right)));
}

constexpr int AccStackSize = 129;
thread_local std::array<nnue::Accumulator, AccStackSize> accStack;

} // namespace

NNUEEngine::NNUEEngine()
    : transposition_(static_cast<std::size_t>(ttSize_) * BucketSize),
      contHistory_(static_cast<std::size_t>(ContHistDim) * ContHistDim, 0),
      rng_(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())) {
    zobrist::init();
    threads_ = defaultThreadCount();
    for (int d = 0; d < MaxLMRDepth; ++d)
        for (int m = 0; m < MaxLMRMoves; ++m)
            lmrTable_[d][m] = (d > 0 && m > 0) ? static_cast<int>(0.75 + std::log(d) * std::log(m) / 2.25) : 0;
}

int NNUEEngine::eval(const Board& board, int ply) const {
    if (ply >= 0 && ply < AccStackSize) {
        auto& acc = accStack[ply];
        if (!acc.computed) nnue_.computeAccumulatorFull(board, acc);
        int score = nnue_.evaluateFromAccumulator(acc, board.side);
#ifndef NDEBUG
        int fullScore = nnue_.evaluate(board, board.side);
        if (acc.computed && std::abs(score - fullScore) > 1) {
            std::cerr << "NNUE mismatch at ply " << ply << ": inc=" << score << " full=" << fullScore << std::endl;
        }
#endif
        return score;
    }
    return nnue_.evaluate(board, board.side);
}

void NNUEEngine::workerSearch(const Board& board, const MoveList& legal,
                               int threadId) const {
    constexpr int SkipSize[]  = {1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4};
    constexpr int SkipPhase[] = {0,0,0,1,0,1,0,1,2,0,1,2,0,1,2,3,0,1,2,3};
    const int si = threadId % 20;

    accStack[0].computed = false;
    nnue_.computeAccumulatorFull(board, accStack[0]);

    MoveList moves = legal;
    const int maxDepth = depthLimit();
    const int pw = rootPruneWidth_;

    for (int depth = 1; depth <= maxDepth && !shouldStop(); ++depth) {
        if (depth > 1 && (depth + SkipPhase[si]) % SkipSize[si] != 0)
            continue;

        std::vector<int> scores(moves.size(), std::numeric_limits<int>::min());
        for (int i = 0; i < static_cast<int>(moves.size()) && !shouldStop(); ++i) {
            auto delta = nnue_.computeMoveDelta(board, moves[i]);
            Board next = board;
            applyMove(next, moves[i]);
            nnue_.updateAccumulatorIncremental(next, accStack[0], delta, accStack[1]);
            int sd = depth - 1;
            if (pw > 0 && depth >= 3 && i >= pw) sd = std::max(0, depth - 3);
            scores[i] = -search(next, sd, 1, -MateScore, MateScore, true, moves[i]);
        }

        if (shouldStop()) break;
        struct IS { int idx; int sc; };
        std::vector<IS> ranked(moves.size());
        for (int i = 0; i < static_cast<int>(moves.size()); ++i) ranked[i] = {i, scores[i]};
        std::stable_sort(ranked.begin(), ranked.end(), [](const IS& a, const IS& b) { return a.sc > b.sc; });
        MoveList reordered;
        for (const auto& r : ranked) reordered.push_back(moves[r.idx]);
        moves = reordered;
    }
}

Move NNUEEngine::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move NNUEEngine::chooseMove(const Board& board, const SearchLimits& limits) {
    return chooseMove(board, limits, InfoCallback{});
}

Move NNUEEngine::chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback) {
    const auto searchStart = std::chrono::steady_clock::now();
    stopped_.store(false);
    nodes_.store(0);
    ++ttGeneration_;
    const int requestedMoveTime = limits.moveTimeMs > 0 ? limits.moveTimeMs : maxMoveTimeMs_;
    const int moveTime = std::clamp(requestedMoveTime, 50, 600000);
    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(moveTime);

    std::cout << "info string params: " << nnuePath_ << " (" << fileModTime(nnuePath_) << ")" << std::endl;
    if (warnOnNoWeights_ && !nnue_.loaded()) {
        if (!fileExists(nnuePath_))
            std::cout << "info string WARNING: " << nnuePath_ << " not found -- using random weights" << std::endl;
        else
            std::cout << "info string ERROR: " << nnuePath_ << " format error -- using random weights" << std::endl;
    }

    nnue_.computeAccumulatorFull(board, accStack[0]);

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        SearchInfo info{};
        info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
        setLastSearchInfo(info);
        return Move{};
    }

    const Color rootSide = board.side;

    if (bookEnabled_ && !book_.empty() && board.moveNumber <= 8) {
        const std::string bookUsi = book_.selectMove(board.hash, rng_);
        if (!bookUsi.empty()) {
            Move bookMove = parseUsiMove(board, bookUsi);
            bool isLegal = false;
            for (const Move& m : legal) {
                if (sameMove(m, bookMove)) { bookMove = m; isLegal = true; break; }
            }
            if (isLegal) {
                SearchInfo info{};
                info.bestMove = bookMove;
                info.hasBestMove = true;
                info.pv.push_back(bookMove);
                setLastSearchInfo(info);
                if (infoCallback) infoCallback(info);
                return bookMove;
            }
        }
    }

    {
        const int mateBudget = std::min(moveTime / 10, 300);
        MateResult mateResult = mateSolver_.searchMate(board, rootSide, 31, mateBudget);
        if (mateResult.found) {
            SearchInfo info;
            info.depth = mateResult.moves;
            info.scoreCp = MateScore;
            info.nodes = mateResult.nodes;
            info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
            info.bestMove = mateResult.bestMove;
            info.hasBestMove = true;
            info.pv = mateResult.pv;
            info.isMate = true;
            info.mateInMoves = mateResult.moves;
            setLastSearchInfo(info);
            if (infoCallback) infoCallback(info);
            return mateResult.bestMove;
        }
    }

    {
        const Color opponent = (rootSide == Black) ? White : Black;
        MateResult tsumero = mateSolver_.detectTsumero(board, opponent, 7);
        if (tsumero.found) {
            std::cout << "info string tsumero detected -- extending search time" << std::endl;
            deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(moveTime * 3 / 2);
        }
    }

    orderMoves(board, legal, 0);
    clearSearchTables();

    std::vector<std::thread> helpers;
    if (threads_ > 1 && legal.size() > 1) {
        for (int t = 1; t < threads_; ++t) {
            helpers.emplace_back([this, &board, &legal, t]() {
                workerSearch(board, legal, t);
            });
        }
    }

    int completedDepth = 0;
    int bestScore = std::numeric_limits<int>::min();
    int prevIterScore = 0;
    std::vector<Move> bestMoves;
    const int maxDepth = depthLimit();
    const int pruneWidth = rootPruneWidth_;

    for (int depth = 1; depth <= maxDepth && !shouldStop(); ++depth) {
        int aspirationAlpha = -MateScore;
        int aspirationBeta = MateScore;
        if (depth >= 2 && std::abs(prevIterScore) < MateScore / 2) {
            aspirationAlpha = prevIterScore - aspirationWindow_;
            aspirationBeta = prevIterScore + aspirationWindow_;
        }

        std::vector<int> scores(legal.size(), std::numeric_limits<int>::min());
        {
            int runningAlpha = aspirationAlpha;
            for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
                if (shouldStop()) break;
                auto delta = nnue_.computeMoveDelta(board, legal[i]);
                Board next = board;
                applyMove(next, legal[i]);
                nnue_.updateAccumulatorIncremental(next, accStack[0], delta, accStack[1]);
                int sd = depth - 1;
                if (pruneWidth > 0 && depth >= 3 && i >= pruneWidth) sd = std::max(0, depth - 3);
                int val = -search(next, sd, 1, -aspirationBeta, -runningAlpha, true, legal[i]);
                scores[i] = val;
                runningAlpha = std::max(runningAlpha, val);
            }
        }

        if (!shouldStop() && depth >= 2 && std::abs(prevIterScore) < MateScore / 2) {
            int depthBest = std::numeric_limits<int>::min();
            for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
                if (scores[i] != std::numeric_limits<int>::min()) depthBest = std::max(depthBest, scores[i]);
            }
            if (depthBest != std::numeric_limits<int>::min() && (depthBest <= aspirationAlpha || depthBest >= aspirationBeta)) {
                int runningAlpha = -MateScore;
                for (int i = 0; i < static_cast<int>(legal.size()) && !shouldStop(); ++i) {
                    auto delta = nnue_.computeMoveDelta(board, legal[i]);
                    Board next = board;
                    applyMove(next, legal[i]);
                    nnue_.updateAccumulatorIncremental(next, accStack[0], delta, accStack[1]);
                    int sd = depth - 1;
                    if (pruneWidth > 0 && depth >= 3 && i >= pruneWidth) sd = std::max(0, depth - 3);
                    int val = -search(next, sd, 1, -MateScore, -runningAlpha, true, legal[i]);
                    scores[i] = val;
                    runningAlpha = std::max(runningAlpha, val);
                }
            }
        }

        if (shouldStop()) break;

        int depthBestScore = std::numeric_limits<int>::min();
        std::vector<Move> depthBestMoves;
        for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
            if (scores[i] == std::numeric_limits<int>::min()) continue;
            if (scores[i] > depthBestScore) {
                depthBestScore = scores[i];
                depthBestMoves.clear();
                depthBestMoves.push_back(legal[i]);
            } else if (scores[i] == depthBestScore) {
                depthBestMoves.push_back(legal[i]);
            }
        }
        if (!depthBestMoves.empty()) {
            completedDepth = depth;
            bestScore = depthBestScore;
            bestMoves = depthBestMoves;

            SearchInfo info;
            info.depth = completedDepth;
            info.scoreCp = std::clamp(bestScore, -MateScore, MateScore);
            info.nodes = nodes_.load();
            info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
            info.bestMove = bestMoves.front();
            info.hasBestMove = true;
            info.pv.push_back(bestMoves.front());
            Board pvBoard = board;
            applyMove(pvBoard, bestMoves.front());
            auto pvTail = extractPV(pvBoard, completedDepth);
            info.pv.insert(info.pv.end(), pvTail.begin(), pvTail.end());
            if (std::abs(depthBestScore) >= MateScore) {
                info.isMate = true;
                const int distance = std::abs(depthBestScore) - MateScore;
                info.mateInMoves = (distance + 1) / 2;
                if (depthBestScore < 0) info.mateInMoves = -info.mateInMoves;
            }
            setLastSearchInfo(info);
            if (infoCallback) infoCallback(info);

            struct IndexedScore { int index; int score; };
            std::vector<IndexedScore> ranked(legal.size());
            for (int i = 0; i < static_cast<int>(legal.size()); ++i) ranked[i] = {i, scores[i]};
            std::stable_sort(ranked.begin(), ranked.end(), [](const IndexedScore& a, const IndexedScore& b) { return a.score > b.score; });
            MoveList reordered;
            for (int i = 0; i < static_cast<int>(ranked.size()); ++i) reordered.push_back(legal[ranked[i].index]);
            legal = reordered;
            prevIterScore = depthBestScore;
        }
    }

    stopped_.store(true);
    for (auto& h : helpers) if (h.joinable()) h.join();

    if (bestMoves.empty()) bestMoves.push_back(legal.front());
    std::uniform_int_distribution<std::size_t> dist(0, bestMoves.size() - 1);
    Move selected = bestMoves[dist(rng_)];
    {
        bool legalFound = false;
        for (const Move& m : legal) { if (sameMove(m, selected)) { legalFound = true; break; } }
        if (legalFound) {
            Board tmp = board;
            applyMove(tmp, selected);
            if (isKingAttacked(tmp, opposite(tmp.side))) legalFound = false;
        }
        if (!legalFound) {
            if (!legal.empty()) selected = legal.front();
            else selected = Move{};
        }
    }
    SearchInfo info;
    info.depth = completedDepth;
    info.scoreCp = bestScore == std::numeric_limits<int>::min() ? 0 : std::clamp(bestScore, -MateScore, MateScore);
    info.nodes = nodes_.load();
    info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
    info.bestMove = selected;
    info.hasBestMove = true;
    setLastSearchInfo(info);
    return selected;
}

// NegaMax search: always returns score from the perspective of board.side
int NNUEEngine::search(Board& board, int depth, int ply, int alpha, int beta, bool allowNullMove, const Move& prevMove, const Move& excludedMove) const {
    nodes_.fetch_add(1);
    if (shouldStop()) return eval(board, ply);

    const int alphaOriginal = alpha;
    const std::uint64_t key = boardHash(board);
    const int ttIndex = static_cast<int>(key & ttMask_);
    const int lockIndex = static_cast<int>((key >> ttBits_) % LockCount);
    Move ttMove{};
    int ttScore = 0, ttDepth = -1;
    std::uint8_t ttFlag = 0;
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const int bucketBase = ttIndex * BucketSize;
        for (int b = 0; b < BucketSize; ++b) {
            const TranspositionEntry& slot = transposition_[bucketBase + b];
            if (slot.key == key && isRecentGeneration(slot.generation)) {
                ttMove = slot.bestMove;
                ttScore = slot.score;
                ttDepth = slot.depth;
                ttFlag = slot.flag;
                if (slot.depth >= depth && excludedMove.to < 0) {
                    if (slot.flag == ExactScore) return slot.score;
                    if (slot.flag == LowerBound) alpha = std::max(alpha, slot.score);
                    else if (slot.flag == UpperBound) beta = std::min(beta, slot.score);
                    if (alpha >= beta) return slot.score;
                }
                break;
            }
        }
    }

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side))
            return -MateScore - depth;
        return 0;
    }

    if (depth <= 0) return quiescence(board, qDepth_, ply, alpha, beta);

    const bool inCheck = isKingAttacked(board, board.side);

    // Null move pruning
    if (allowNullMove && !inCheck && depth >= nmpMinDepth_ && ply > 0) {
        if (ply + 1 < AccStackSize) accStack[ply + 1] = accStack[ply];
        NullMoveUndoInfo nullUndo;
        applyNullMove(board, nullUndo);
        const int nullVal = -search(board, depth - 1 - nmpReduction_, ply + 1, -beta, -beta + 1, false, Move{});
        undoNullMove(board, nullUndo);
        if (nullVal >= beta) return nullVal;
    }

    int staticEval = eval(board, ply);

    // Reverse futility pruning
    if (!inCheck && depth <= 3 && ply > 0 && std::abs(alpha) < MateScore / 2 && std::abs(beta) < MateScore / 2) {
        const int rfpMargin = depth * 200;
        if (staticEval - rfpMargin >= beta) return staticEval;
    }

    // Razoring
    if (!inCheck && depth <= 2 && ply > 0 && std::abs(alpha) < MateScore / 2 && std::abs(beta) < MateScore / 2) {
        const int razorMargin = 300 + depth * 200;
        if (staticEval + razorMargin <= alpha)
            return quiescence(board, qDepth_, ply, alpha, beta);
    }

    // IID
    if (ttMove.to < 0 && depth >= iidMinDepth_ && !inCheck) {
        search(board, depth - 3, ply, alpha, beta, false, prevMove);
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const int iidBase = ttIndex * BucketSize;
        for (int b = 0; b < BucketSize; ++b) {
            const TranspositionEntry& slot = transposition_[iidBase + b];
            if (slot.key == key && isRecentGeneration(slot.generation)) { ttMove = slot.bestMove; break; }
        }
    }

    // Singular extension
    int singularExtension = 0;
    if (depth >= seMinDepth_ && ttMove.to >= 0 && excludedMove.to < 0
        && ttDepth >= depth - 3 && ttFlag != UpperBound
        && std::abs(ttScore) < MateScore / 2) {
        const int seBeta = ttScore - 2 * depth;
        int seVal = search(board, depth / 2, ply, seBeta - 1, seBeta, false, prevMove, ttMove);
        if (seVal < seBeta) singularExtension = 1;
    }

    orderMoves(board, legal, ply, prevMove);

    const bool canFutilityPrune = !inCheck && (depth == 1 || depth == 2);
    const int futilityMargin = depth == 1 ? futilityMargin1_ : futilityMargin2_;
    const int lmpThreshold = 3 + depth * depth;

    int best = std::numeric_limits<int>::min();
    Move bestMoveLocal{};
    Move quietsTried[MoveList::Capacity];
    int quietsCount = 0;
    int moveIndex = 0;
    bool anySearched = false;

    for (const Move& move : legal) {
        if (excludedMove.to >= 0 && sameMove(move, excludedMove)) { ++moveIndex; continue; }
        const bool isQuiet = (move.isDrop() || board.squares[move.to] == 0) && !move.promote;
        const bool isCheck = givesCheck(board, move);
        if (canFutilityPrune && isQuiet && !isCheck && staticEval + futilityMargin <= alpha) { ++moveIndex; continue; }
        if (depth <= 3 && moveIndex >= lmpThreshold && isQuiet && !isCheck && !inCheck && anySearched) { ++moveIndex; continue; }
        if (!isQuiet && !isCheck && !inCheck && anySearched && depth <= 4 && staticExchangeEval(board, move) < -100 * depth) { ++moveIndex; continue; }

        auto delta = nnue_.computeMoveDelta(board, move);
        UndoInfo undo;
        applyMove(board, move, undo);
        if (ply + 1 < AccStackSize) nnue_.updateAccumulatorIncremental(board, accStack[ply], delta, accStack[ply + 1]);
        const bool givesCheckNow = isKingAttacked(board, board.side);
        int extension = (givesCheckNow && ply < MaxPly - 10) ? 1 : 0;
        if (singularExtension > 0 && sameMove(move, ttMove)) extension = std::max(extension, singularExtension);
        const int newDepth = depth - 1 + extension;

        int val = 0;
        if (moveIndex == 0) {
            val = -search(board, newDepth, ply + 1, -beta, -alpha, true, move);
        } else {
            bool needsFullWindow = false;
            int R = lmrTable_[std::min(depth, MaxLMRDepth - 1)][std::min(moveIndex, MaxLMRMoves - 1)];
            if (R > 0 && isQuiet && !isCheck && !inCheck && !givesCheckNow) {
                const int cIdx = board.side == Black ? 0 : 1;
                int histVal = 0;
                if (move.isDrop()) {
                    int dti = static_cast<int>(move.drop) - 1;
                    if (dti >= 0 && dti < 7) histVal = dropHistory_[cIdx][dti][move.to];
                } else if (move.from >= 0 && move.from < BoardSize) {
                    histVal = history_[cIdx][move.from][move.to];
                }
                if (histVal > 2000) R = std::max(0, R - 1);
                else if (histVal < -2000) R += 1;
            }
            if (R > 0 && isQuiet && !isCheck && !inCheck && !givesCheckNow) {
                val = -search(board, std::max(1, newDepth - R), ply + 1, -alpha - 1, -alpha, true, move);
                if (val > alpha) {
                    val = -search(board, newDepth, ply + 1, -alpha - 1, -alpha, true, move);
                    if (val > alpha && val < beta) needsFullWindow = true;
                }
            } else {
                val = -search(board, newDepth, ply + 1, -alpha - 1, -alpha, true, move);
                if (val > alpha && val < beta) needsFullWindow = true;
            }
            if (needsFullWindow) val = -search(board, newDepth, ply + 1, -beta, -alpha, true, move);
        }

        undoMove(board, move, undo);
        anySearched = true;
        if (val > best) { best = val; bestMoveLocal = move; }
        alpha = std::max(alpha, best);
        if (alpha >= beta) {
            const int cIdx = board.side == Black ? 0 : 1;
            auto getPt = [&](const Move& m) -> PieceType {
                if (m.isDrop()) return m.drop;
                if (m.from >= 0 && m.from < BoardSize) return typeOf(board.squares[m.from]);
                return static_cast<PieceType>(0);
            };
            if (isQuiet) {
                storeKiller(ply, move);
                updateHistory(board.side, move, depth, true);
                storeCounterMove(board.side, prevMove, move);
                if (move.isDrop()) {
                    int dti = static_cast<int>(move.drop) - 1;
                    if (dti >= 0 && dti < 7) gravityUpdate(dropHistory_[cIdx][dti][move.to], depth, true);
                }
                if (prevMove.to >= 0 && prevMove.to < BoardSize) {
                    PieceType prevPt = prevMove.isDrop() ? prevMove.drop : typeOf(board.squares[prevMove.to]);
                    PieceType curPt = getPt(move);
                    int pi = static_cast<int>(prevPt) - 1, ci = static_cast<int>(curPt) - 1;
                    if (pi >= 0 && pi < ContHistPieces && ci >= 0 && ci < ContHistPieces) {
                        int idx = (pi * BoardSize + prevMove.to) * ContHistDim + ci * BoardSize + move.to;
                        gravityUpdate(contHistory_[idx], depth, true);
                    }
                }
                for (int q = 0; q < quietsCount; ++q) {
                    updateHistory(board.side, quietsTried[q], depth, false);
                    if (quietsTried[q].isDrop()) {
                        int dti = static_cast<int>(quietsTried[q].drop) - 1;
                        if (dti >= 0 && dti < 7) gravityUpdate(dropHistory_[cIdx][dti][quietsTried[q].to], depth, false);
                    }
                    if (prevMove.to >= 0 && prevMove.to < BoardSize) {
                        PieceType prevPt = prevMove.isDrop() ? prevMove.drop : typeOf(board.squares[prevMove.to]);
                        PieceType curPt = getPt(quietsTried[q]);
                        int pi = static_cast<int>(prevPt) - 1, ci = static_cast<int>(curPt) - 1;
                        if (pi >= 0 && pi < ContHistPieces && ci >= 0 && ci < ContHistPieces) {
                            int idx = (pi * BoardSize + prevMove.to) * ContHistDim + ci * BoardSize + quietsTried[q].to;
                            gravityUpdate(contHistory_[idx], depth, false);
                        }
                    }
                }
            } else {
                if (!move.isDrop() && move.from >= 0 && move.from < BoardSize && board.squares[move.to] != 0) {
                    int mpi = static_cast<int>(typeOf(board.squares[move.from])) - 1;
                    int cpi = static_cast<int>(typeOf(board.squares[move.to])) - 1;
                    if (mpi >= 0 && mpi < ContHistPieces && cpi >= 0 && cpi < ContHistPieces)
                        gravityUpdate(captHistory_[mpi][move.to][cpi], depth, true);
                }
            }
            break;
        }
        if (isQuiet && quietsCount < MoveList::Capacity) quietsTried[quietsCount++] = move;
        ++moveIndex;
    }

    if (!anySearched) return canFutilityPrune ? staticEval : -MateScore - depth;

    if (excludedMove.to < 0) {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const int storeBase = ttIndex * BucketSize;
        int replaceIdx = 0;
        int worstQuality = std::numeric_limits<int>::max();
        for (int b = 0; b < BucketSize; ++b) {
            const TranspositionEntry& s = transposition_[storeBase + b];
            if (s.key == key) { replaceIdx = b; worstQuality = -1; break; }
            if (s.depth < 0) { replaceIdx = b; worstQuality = -1; break; }
            int q = s.depth * 8 + (isRecentGeneration(s.generation) ? 256 : 0);
            if (q < worstQuality) { worstQuality = q; replaceIdx = b; }
        }
        TranspositionEntry& slot = transposition_[storeBase + replaceIdx];
        if (slot.key != key || slot.generation != ttGeneration_ || depth >= slot.depth) {
            slot.key = key; slot.depth = depth; slot.score = best; slot.bestMove = bestMoveLocal;
            slot.staticEval = staticEval;
            slot.flag = best <= alphaOriginal ? UpperBound : (best >= beta ? LowerBound : ExactScore);
            slot.generation = ttGeneration_;
        }
    }
    return best;
}

// NegaMax quiescence: returns score from the perspective of board.side
int NNUEEngine::quiescence(Board& board, int depth, int ply, int alpha, int beta) const {
    nodes_.fetch_add(1);
    if (shouldStop()) return eval(board, ply);

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side))
            return -MateScore - depth;
        return 0;
    }

    int standPat = eval(board, ply);
    if (depth <= 0) return standPat;

    const bool inCheck = isKingAttacked(board, board.side);
    orderMoves(board, legal, ply);

    if (standPat >= beta) return standPat;
    if (!inCheck && standPat + deltaMargin_ <= alpha) return standPat;
    alpha = std::max(alpha, standPat);
    int best = standPat;

    for (const Move& move : legal) {
        if (!inCheck) {
            const bool capture = !move.isDrop() && board.squares[move.to] != 0;
            const bool promotion = move.promote;
            if (!capture && !promotion) {
                if (depth < qCheckDepthMin_ || !givesCheck(board, move)) continue;
            }
            if (capture && staticExchangeEval(board, move) < 0) continue;
        }
        auto delta = nnue_.computeMoveDelta(board, move);
        UndoInfo undo;
        applyMove(board, move, undo);
        if (ply + 1 < AccStackSize) nnue_.updateAccumulatorIncremental(board, accStack[ply], delta, accStack[ply + 1]);
        int val = -quiescence(board, depth - 1, ply + 1, -beta, -alpha);
        undoMove(board, move, undo);
        if (val > best) best = val;
        alpha = std::max(alpha, best);
        if (alpha >= beta) break;
    }
    return best;
}

void NNUEEngine::orderMoves(const Board& board, MoveList& moves, int ply, const Move& prevMove) const {
    const std::uint64_t key = boardHash(board);
    const int ttIndex = static_cast<int>(key & ttMask_);
    const int lockIndex = static_cast<int>((key >> ttBits_) % LockCount);
    Move ttMove{};
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const int bucketBase = ttIndex * BucketSize;
        for (int b = 0; b < BucketSize; ++b) {
            const TranspositionEntry& slot = transposition_[bucketBase + b];
            if (slot.key == key && isRecentGeneration(slot.generation)) { ttMove = slot.bestMove; break; }
        }
    }
    struct ScoredMove { Move move; int score; };
    const int n = moves.size();
    ScoredMove scored[MoveList::Capacity];
    for (int i = 0; i < n; ++i) scored[i] = {moves[i], moveOrderScore(board, moves[i], ply, ttMove, prevMove)};
    std::stable_sort(scored, scored + n, [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });
    for (int i = 0; i < n; ++i) moves[i] = scored[i].move;
}

int NNUEEngine::moveOrderScore(const Board& board, const Move& move, int ply, const Move& ttMove, const Move& prevMove) const {
    if (ttMove.to >= 0 && sameMove(move, ttMove)) return 20000;

    int score = 0;
    if (!move.isDrop() && board.squares[move.to] != 0) {
        const PieceType captured = typeOf(board.squares[move.to]);
        const PieceType moving = typeOf(board.squares[move.from]);
        score += 10000 + pieceValue(captured) * 10 - pieceValue(moving);
        int mpi = static_cast<int>(moving) - 1, cpi = static_cast<int>(captured) - 1;
        if (mpi >= 0 && mpi < ContHistPieces && cpi >= 0 && cpi < ContHistPieces)
            score += captHistory_[mpi][move.to][cpi] / 8;
    }
    if (givesCheck(board, move)) score += 8000;
    if (move.promote) score += 6000;
    if (score >= 6000) return score;

    if (ply < MaxPly) {
        if (sameMove(move, killers_[ply][0])) return 5000;
        if (sameMove(move, killers_[ply][1])) return 4900;
    }
    if (prevMove.to >= 0) {
        const int colorIdx = board.side == Black ? 0 : 1;
        const int prevFrom = prevMove.isDrop() ? prevMove.to : prevMove.from;
        if (prevFrom >= 0 && prevFrom < BoardSize && prevMove.to >= 0 && prevMove.to < BoardSize) {
            if (sameMove(move, counterMoves_[colorIdx][prevFrom][prevMove.to])) return 4800;
        }
    }
    if (move.isDrop()) {
        score += pieceValue(move.drop) / 3;
        const int colorIdx2 = board.side == Black ? 0 : 1;
        int dti = static_cast<int>(move.drop) - 1;
        if (dti >= 0 && dti < 7) score += dropHistory_[colorIdx2][dti][move.to];
        const int enemyKing = board.side == Black ? board.whiteKingSquare : board.blackKingSquare;
        if (enemyKing >= 0 && chebyshevDistance(move.to, enemyKing) <= 2) score += 200;
    } else if (board.squares[move.to] == 0) {
        const int colorIdx = board.side == Black ? 0 : 1;
        if (move.from >= 0 && move.from < BoardSize) score += history_[colorIdx][move.from][move.to];
        const Color enemy = opposite(board.side);
        if (isSquareAttacked(board, move.to, enemy)) {
            const int enemyKing = board.side == Black ? board.whiteKingSquare : board.blackKingSquare;
            if (enemyKing >= 0 && chebyshevDistance(move.to, enemyKing) <= 2) score += 300;
        }
    }
    if (prevMove.to >= 0 && prevMove.to < BoardSize && score < 6000) {
        PieceType prevPt = prevMove.isDrop() ? prevMove.drop : typeOf(board.squares[prevMove.to]);
        PieceType curPt = move.isDrop() ? move.drop
            : (move.from >= 0 && move.from < BoardSize ? typeOf(board.squares[move.from]) : static_cast<PieceType>(0));
        int pi = static_cast<int>(prevPt) - 1, ci = static_cast<int>(curPt) - 1;
        if (pi >= 0 && pi < ContHistPieces && ci >= 0 && ci < ContHistPieces) {
            int idx = (pi * BoardSize + prevMove.to) * ContHistDim + ci * BoardSize + move.to;
            score += contHistory_[idx] / 2;
        }
    }
    return score;
}

void NNUEEngine::storeKiller(int ply, const Move& move) const {
    if (ply >= MaxPly) return;
    if (sameMove(move, killers_[ply][0])) return;
    killers_[ply][1] = killers_[ply][0];
    killers_[ply][0] = move;
}

void NNUEEngine::updateHistory(Color side, const Move& move, int depth, bool good) const {
    if (move.isDrop() || move.promote) return;
    if (move.from < 0 || move.from >= BoardSize || move.to < 0 || move.to >= BoardSize) return;
    const int colorIdx = side == Black ? 0 : 1;
    const int bonus = good ? depth * depth : -(depth * depth);
    int entry = history_[colorIdx][move.from][move.to];
    entry += bonus - entry * std::abs(bonus) / 16384;
    history_[colorIdx][move.from][move.to] = static_cast<std::int16_t>(std::clamp(entry, -16384, 16384));
}

void NNUEEngine::storeCounterMove(Color side, const Move& prevMove, const Move& counterMove) const {
    if (prevMove.to < 0) return;
    const int colorIdx = side == Black ? 0 : 1;
    const int from = prevMove.isDrop() ? prevMove.to : prevMove.from;
    if (from < 0 || from >= BoardSize || prevMove.to < 0 || prevMove.to >= BoardSize) return;
    counterMoves_[colorIdx][from][prevMove.to] = counterMove;
}

void NNUEEngine::clearSearchTables() const {
    for (auto& ply : killers_) for (auto& slot : ply) slot = Move{};
    for (auto& color : history_) for (auto& from : color) for (auto& val : from) val /= 2;
    for (auto& color : counterMoves_) for (auto& from : color) for (auto& m : from) m = Move{};
    for (auto& v : contHistory_) v /= 2;
    for (auto& a : captHistory_) for (auto& b : a) for (auto& v : b) v /= 2;
    for (auto& a : dropHistory_) for (auto& b : a) for (auto& v : b) v /= 2;
}

bool NNUEEngine::shouldStop() const {
    if (stopped_.load()) return true;
    if (std::chrono::steady_clock::now() >= deadline_) { stopped_.store(true); return true; }
    return false;
}

std::uint64_t NNUEEngine::boardHash(const Board& board) const {
    return board.hash ^ (board.side == Black ? 0x9E3779B97F4A7C15ULL : 0x6C62272E07BB0142ULL);
}

void NNUEEngine::setLastSearchInfo(const SearchInfo& info) const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    lastSearchInfo_ = info;
}

SearchInfo NNUEEngine::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    return lastSearchInfo_;
}

std::vector<Move> NNUEEngine::extractPV(Board board, int maxDepth) const {
    std::vector<Move> pv;
    for (int i = 0; i < maxDepth; ++i) {
        const std::uint64_t key = boardHash(board);
        const int ttIndex = static_cast<int>(key & ttMask_);
        const int lockIndex = static_cast<int>((key >> ttBits_) % LockCount);
        Move ttMove;
        {
            std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
            const int bucketBase = ttIndex * BucketSize;
            bool found = false;
            for (int b = 0; b < BucketSize; ++b) {
                const TranspositionEntry& slot = transposition_[bucketBase + b];
                if (slot.key == key && isRecentGeneration(slot.generation)) {
                    ttMove = slot.bestMove; found = true; break;
                }
            }
            if (!found) break;
        }
        if (ttMove.to < 0) break;
        pv.push_back(ttMove);
        applyMove(board, ttMove);
    }
    return pv;
}

int NNUEEngine::depthLimit() const {
    return searchDepth_ > 0 ? searchDepth_ : 128;
}

MateResult NNUEEngine::searchMate(const Board& board, int timeLimitMs) {
    return mateSolver_.searchMate(board, board.side, 31, timeLimitMs);
}

void NNUEEngine::setHashSizeMB(int mb) {
    mb = std::clamp(mb, 1, 65536);
    std::size_t bytes = static_cast<std::size_t>(mb) * 1024 * 1024;
    std::size_t entrySize = sizeof(TranspositionEntry);
    std::size_t numBuckets = bytes / (entrySize * BucketSize);
    int bits = 1;
    while ((1ULL << (bits + 1)) <= numBuckets) ++bits;
    ttBits_ = bits;
    ttSize_ = 1 << ttBits_;
    ttMask_ = ttSize_ - 1;
    transposition_.assign(static_cast<std::size_t>(ttSize_) * BucketSize, TranspositionEntry{});
}

void NNUEEngine::setSearchDepth(int depth) { searchDepth_ = std::clamp(depth, 0, 128); }
void NNUEEngine::setMaxMoveTimeMs(int ms) { maxMoveTimeMs_ = std::clamp(ms, 50, 600000); }
void NNUEEngine::setThreads(int threads) { threads_ = std::clamp(threads, 1, 256); }
int NNUEEngine::threadCount() const { return threads_; }
void NNUEEngine::setBookEnabled(bool enabled) { bookEnabled_ = enabled; }
bool NNUEEngine::loadBook(const std::string& path) { return book_.load(path); }
bool NNUEEngine::loadNNUE(const std::string& path) { nnuePath_ = path; return nnue_.load(path); }

void NNUEEngine::setParam(const std::string& name, int value) {
    if (name == "SEMinDepth") seMinDepth_ = value;
    else if (name == "NMPMinDepth") nmpMinDepth_ = value;
    else if (name == "NMPReduction") nmpReduction_ = value;
    else if (name == "FutilityMargin1") futilityMargin1_ = value;
    else if (name == "FutilityMargin2") futilityMargin2_ = value;
    else if (name == "AspirationWindow") aspirationWindow_ = value;
    else if (name == "IIDMinDepth") iidMinDepth_ = value;
    else if (name == "DeltaMargin") deltaMargin_ = value;
    else if (name == "QDepth") qDepth_ = value;
    else if (name == "QCheckDepthMin") qCheckDepthMin_ = value;
    else if (name == "RootPruneWidth") rootPruneWidth_ = value;
}

int NNUEEngine::getParam(const std::string& name) const {
    if (name == "SEMinDepth") return seMinDepth_;
    if (name == "NMPMinDepth") return nmpMinDepth_;
    if (name == "NMPReduction") return nmpReduction_;
    if (name == "FutilityMargin1") return futilityMargin1_;
    if (name == "FutilityMargin2") return futilityMargin2_;
    if (name == "AspirationWindow") return aspirationWindow_;
    if (name == "IIDMinDepth") return iidMinDepth_;
    if (name == "DeltaMargin") return deltaMargin_;
    if (name == "QDepth") return qDepth_;
    if (name == "QCheckDepthMin") return qCheckDepthMin_;
    if (name == "RootPruneWidth") return rootPruneWidth_;
    return 0;
}

} // namespace shogi
