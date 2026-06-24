#include "nnue_engine.h"

#include "movegen.h"
#include "notation.h"
#include "position.h"

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
    : transposition_(TTSize),
      rng_(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())) {
    zobrist::init();
    threads_ = defaultThreadCount();
}

int NNUEEngine::eval(const Board& board, Color rootSide, int ply) const {
    if (ply >= 0 && ply < AccStackSize) {
        auto& acc = accStack[ply];
        if (!acc.computed) nnue_.computeAccumulatorFull(board, acc);
        int incScore = nnue_.evaluateFromAccumulator(acc, rootSide);
#ifndef NDEBUG
        int fullScore = nnue_.evaluate(board, rootSide);
        if (acc.computed && std::abs(incScore - fullScore) > 1) {
            std::cerr << "NNUE mismatch at ply " << ply << ": inc=" << incScore << " full=" << fullScore << std::endl;
        }
#endif
        return incScore;
    }
    return nnue_.evaluate(board, rootSide);
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

    orderMoves(board, legal, rootSide, 0);
    clearSearchTables();

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

        // Score root moves
        std::vector<int> scores(legal.size(), std::numeric_limits<int>::min());
        {
            std::atomic<int> sharedAlpha{aspirationAlpha};
            const int workerCount = std::min<int>(std::max(1, threads_), legal.size());

            if (workerCount <= 1) {
                int runningAlpha = aspirationAlpha;
                for (int i = 0; i < legal.size(); ++i) {
                    if (shouldStop()) break;
                    auto delta = nnue_.computeMoveDelta(board, legal[i]);
                    Board next = board;
                    applyMove(next, legal[i]);
                    nnue_.updateAccumulatorIncremental(accStack[0], delta, accStack[1]);
                    int sd = depth - 1;
                    if (pruneWidth > 0 && depth >= 3 && i >= pruneWidth) sd = std::max(0, depth - 3);
                    scores[i] = search(next, sd, 1, runningAlpha, aspirationBeta, rootSide, true, legal[i]);
                    runningAlpha = std::max(runningAlpha, scores[i]);
                }
            } else {
                std::atomic_int nextIndex{0};
                std::vector<std::thread> workers;
                workers.reserve(workerCount);
                for (int w = 0; w < workerCount; ++w) {
                    workers.emplace_back([&]() {
                        accStack[0].computed = false;
                        nnue_.computeAccumulatorFull(board, accStack[0]);
                        while (!shouldStop()) {
                            const int i = nextIndex.fetch_add(1);
                            if (i >= static_cast<int>(legal.size())) break;
                            auto delta = nnue_.computeMoveDelta(board, legal[i]);
                            Board next = board;
                            applyMove(next, legal[i]);
                            nnue_.updateAccumulatorIncremental(accStack[0], delta, accStack[1]);
                            int sd = depth - 1;
                            if (pruneWidth > 0 && depth >= 3 && i >= pruneWidth) sd = std::max(0, depth - 3);
                            const int la = sharedAlpha.load(std::memory_order_relaxed);
                            scores[i] = search(next, sd, 1, la, aspirationBeta, rootSide, true, legal[i]);
                            int expected = sharedAlpha.load(std::memory_order_relaxed);
                            while (scores[i] > expected) {
                                if (sharedAlpha.compare_exchange_weak(expected, scores[i], std::memory_order_relaxed)) break;
                            }
                        }
                    });
                }
                for (auto& w : workers) if (w.joinable()) w.join();
            }
        }

        // Re-search with full window if aspiration failed
        if (!shouldStop() && depth >= 2 && std::abs(prevIterScore) < MateScore / 2) {
            int depthBest = std::numeric_limits<int>::min();
            for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
                if (scores[i] != std::numeric_limits<int>::min()) depthBest = std::max(depthBest, scores[i]);
            }
            if (depthBest != std::numeric_limits<int>::min() && (depthBest <= aspirationAlpha || depthBest >= aspirationBeta)) {
                int runningAlpha = -MateScore;
                for (int i = 0; i < legal.size() && !shouldStop(); ++i) {
                    auto delta = nnue_.computeMoveDelta(board, legal[i]);
                    Board next = board;
                    applyMove(next, legal[i]);
                    nnue_.updateAccumulatorIncremental(accStack[0], delta, accStack[1]);
                    int sd = depth - 1;
                    if (pruneWidth > 0 && depth >= 3 && i >= pruneWidth) sd = std::max(0, depth - 3);
                    scores[i] = search(next, sd, 1, runningAlpha, MateScore, rootSide, true, legal[i]);
                    runningAlpha = std::max(runningAlpha, scores[i]);
                }
            }
        }

        if (shouldStop()) break;

        int depthBestScore = std::numeric_limits<int>::min();
        std::vector<Move> depthBestMoves;
        for (int i = 0; i < legal.size(); ++i) {
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
            auto pvTail = extractPV(pvBoard, rootSide, completedDepth);
            info.pv.insert(info.pv.end(), pvTail.begin(), pvTail.end());
            if (std::abs(depthBestScore) >= MateScore) {
                info.isMate = true;
                const int distance = std::abs(depthBestScore) - MateScore;
                info.mateInMoves = (distance + 1) / 2;
                if (depthBestScore < 0) info.mateInMoves = -info.mateInMoves;
            }
            setLastSearchInfo(info);
            if (infoCallback) infoCallback(info);

            // Reorder root moves
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

int NNUEEngine::search(Board& board, int depth, int ply, int alpha, int beta, Color rootSide, bool allowNullMove, const Move& prevMove) const {
    nodes_.fetch_add(1);
    if (shouldStop()) return eval(board, rootSide, ply);

    const int alphaOriginal = alpha;
    const int betaOriginal = beta;
    const std::uint64_t key = boardHash(board, rootSide);
    const int ttIndex = static_cast<int>(key & TTMask);
    const int lockIndex = static_cast<int>((key >> TTBits) % LockCount);
    Move ttMove{};
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.key == key && slot.generation == ttGeneration_) {
            ttMove = slot.bestMove;
            if (slot.depth >= depth) {
                if (slot.flag == ExactScore) return slot.score;
                if (slot.flag == LowerBound) alpha = std::max(alpha, slot.score);
                else if (slot.flag == UpperBound) beta = std::min(beta, slot.score);
                if (alpha >= beta) return slot.score;
            }
        }
    }

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side))
            return board.side == rootSide ? -MateScore - depth : MateScore + depth;
        return 0;
    }

    if (depth <= 0) return quiescence(board, qDepth_, ply, alpha, beta, rootSide);

    const bool inCheck = isKingAttacked(board, board.side);
    const bool maximizing = board.side == rootSide;

    if (allowNullMove && !inCheck && depth >= nmpMinDepth_ && ply > 0) {
        if (ply + 1 < AccStackSize) accStack[ply + 1] = accStack[ply];
        NullMoveUndoInfo nullUndo;
        applyNullMove(board, nullUndo);
        const int nullVal = search(board, depth - 1 - nmpReduction_, ply + 1, alpha, beta, rootSide, false, Move{});
        undoNullMove(board, nullUndo);
        if (maximizing) { if (nullVal >= beta) return nullVal; }
        else { if (nullVal <= alpha) return nullVal; }
    }

    int staticEval = eval(board, rootSide, ply);

    // Reverse futility pruning
    if (!inCheck && depth <= 3 && ply > 0 && std::abs(alpha) < MateScore / 2 && std::abs(beta) < MateScore / 2) {
        const int rfpMargin = depth * 200;
        if (maximizing && staticEval - rfpMargin >= beta) return staticEval;
        if (!maximizing && staticEval + rfpMargin <= alpha) return staticEval;
    }

    // Razoring
    if (!inCheck && depth <= 2 && ply > 0 && std::abs(alpha) < MateScore / 2 && std::abs(beta) < MateScore / 2) {
        const int razorMargin = 300 + depth * 200;
        if (maximizing && staticEval + razorMargin <= alpha)
            return quiescence(board, qDepth_, ply, alpha, beta, rootSide);
        if (!maximizing && staticEval - razorMargin >= beta)
            return quiescence(board, qDepth_, ply, alpha, beta, rootSide);
    }

    if (ttMove.to < 0 && depth >= iidMinDepth_ && !inCheck) {
        search(board, depth - 3, ply, alpha, beta, rootSide, false, prevMove);
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.key == key && slot.generation == ttGeneration_) ttMove = slot.bestMove;
    }

    orderMoves(board, legal, rootSide, ply, prevMove);

    const bool canFutilityPrune = !inCheck && (depth == 1 || depth == 2);
    const int futilityMargin = depth == 1 ? futilityMargin1_ : futilityMargin2_;
    const int lmpThreshold = 3 + depth * depth;

    if (maximizing) {
        int best = std::numeric_limits<int>::min();
        Move bestMoveLocal{};
        Move quietsTried[MoveList::Capacity];
        int quietsCount = 0;
        int moveIndex = 0;
        bool anySearched = false;

        for (const Move& move : legal) {
            const bool isQuiet = (move.isDrop() || board.squares[move.to] == 0) && !move.promote;
            const bool isCheck = givesCheck(board, move);
            if (canFutilityPrune && isQuiet && !isCheck && staticEval + futilityMargin <= alpha) { ++moveIndex; continue; }
            if (depth <= 3 && moveIndex >= lmpThreshold && isQuiet && !isCheck && !inCheck && anySearched) { ++moveIndex; continue; }

            auto delta = nnue_.computeMoveDelta(board, move);
            UndoInfo undo;
            applyMove(board, move, undo);
            if (ply + 1 < AccStackSize) nnue_.updateAccumulatorIncremental(accStack[ply], delta, accStack[ply + 1]);
            const bool givesCheckNow = isKingAttacked(board, board.side);
            int extension = (givesCheckNow && ply < MaxPly - 10) ? 1 : 0;
            const int newDepth = depth - 1 + extension;

            int val = 0;
            if (moveIndex == 0) {
                val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
            } else {
                bool needsFullWindow = false;
                if (depth >= lmrMinDepth_ && moveIndex >= lmrFullDepthMoves_ && isQuiet && !isCheck && !inCheck && !givesCheckNow) {
                    int R = 1 + (depth >= 6 ? 1 : 0) + (moveIndex >= 10 ? 1 : 0);
                    val = search(board, std::max(1, newDepth - R), ply + 1, alpha, alpha + 1, rootSide, true, move);
                    if (val > alpha) {
                        val = search(board, newDepth, ply + 1, alpha, alpha + 1, rootSide, true, move);
                        if (val > alpha && val < beta) needsFullWindow = true;
                    }
                } else {
                    val = search(board, newDepth, ply + 1, alpha, alpha + 1, rootSide, true, move);
                    if (val > alpha && val < beta) needsFullWindow = true;
                }
                if (needsFullWindow) val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
            }

            undoMove(board, move, undo);
            anySearched = true;
            if (val > best) { best = val; bestMoveLocal = move; }
            alpha = std::max(alpha, best);
            if (alpha >= beta) {
                if (isQuiet) {
                    storeKiller(ply, move);
                    updateHistory(board.side, move, depth, true);
                    storeCounterMove(board.side, prevMove, move);
                    for (int q = 0; q < quietsCount; ++q) updateHistory(board.side, quietsTried[q], depth, false);
                }
                break;
            }
            if (isQuiet && quietsCount < MoveList::Capacity) quietsTried[quietsCount++] = move;
            ++moveIndex;
        }

        if (!anySearched) return canFutilityPrune ? staticEval : (board.side == rootSide ? -MateScore - depth : MateScore + depth);

        {
            std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
            TranspositionEntry& slot = transposition_[ttIndex];
            if (slot.generation != ttGeneration_ || depth >= slot.depth) {
                slot.key = key; slot.depth = depth; slot.score = best; slot.bestMove = bestMoveLocal;
                slot.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
                slot.generation = ttGeneration_;
            }
        }
        return best;
    }

    // Minimizing
    int best = std::numeric_limits<int>::max();
    Move bestMoveLocal{};
    Move quietsTried[MoveList::Capacity];
    int quietsCount = 0;
    int moveIndex = 0;
    bool anySearched = false;

    for (const Move& move : legal) {
        const bool isQuiet = (move.isDrop() || board.squares[move.to] == 0) && !move.promote;
        const bool isCheck = givesCheck(board, move);
        if (canFutilityPrune && isQuiet && !isCheck && staticEval - futilityMargin >= beta) { ++moveIndex; continue; }
        if (depth <= 3 && moveIndex >= lmpThreshold && isQuiet && !isCheck && !inCheck && anySearched) { ++moveIndex; continue; }

        auto delta = nnue_.computeMoveDelta(board, move);
        UndoInfo undo;
        applyMove(board, move, undo);
        if (ply + 1 < AccStackSize) nnue_.updateAccumulatorIncremental(accStack[ply], delta, accStack[ply + 1]);
        const bool givesCheckNow = isKingAttacked(board, board.side);
        int extension = (givesCheckNow && ply < MaxPly - 10) ? 1 : 0;
        const int newDepth = depth - 1 + extension;

        int val = 0;
        if (moveIndex == 0) {
            val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
        } else {
            bool needsFullWindow = false;
            if (depth >= lmrMinDepth_ && moveIndex >= lmrFullDepthMoves_ && isQuiet && !isCheck && !inCheck && !givesCheckNow) {
                int R = 1 + (depth >= 6 ? 1 : 0) + (moveIndex >= 10 ? 1 : 0);
                val = search(board, std::max(1, newDepth - R), ply + 1, beta - 1, beta, rootSide, true, move);
                if (val < beta) {
                    val = search(board, newDepth, ply + 1, beta - 1, beta, rootSide, true, move);
                    if (val < beta && val > alpha) needsFullWindow = true;
                }
            } else {
                val = search(board, newDepth, ply + 1, beta - 1, beta, rootSide, true, move);
                if (val < beta && val > alpha) needsFullWindow = true;
            }
            if (needsFullWindow) val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
        }

        undoMove(board, move, undo);
        anySearched = true;
        if (val < best) { best = val; bestMoveLocal = move; }
        beta = std::min(beta, best);
        if (alpha >= beta) {
            if (isQuiet) {
                storeKiller(ply, move);
                updateHistory(board.side, move, depth, true);
                storeCounterMove(board.side, prevMove, move);
                for (int q = 0; q < quietsCount; ++q) updateHistory(board.side, quietsTried[q], depth, false);
            }
            break;
        }
        if (isQuiet && quietsCount < MoveList::Capacity) quietsTried[quietsCount++] = move;
        ++moveIndex;
    }

    if (!anySearched) return canFutilityPrune ? staticEval : (board.side == rootSide ? -MateScore - depth : MateScore + depth);

    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.generation != ttGeneration_ || depth >= slot.depth) {
            slot.key = key; slot.depth = depth; slot.score = best; slot.bestMove = bestMoveLocal;
            slot.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
            slot.generation = ttGeneration_;
        }
    }
    return best;
}

int NNUEEngine::quiescence(Board& board, int depth, int ply, int alpha, int beta, Color rootSide) const {
    nodes_.fetch_add(1);
    if (shouldStop()) return eval(board, rootSide, ply);

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side))
            return board.side == rootSide ? -MateScore - depth : MateScore + depth;
        return 0;
    }

    int standPat = eval(board, rootSide, ply);
    if (depth <= 0) return standPat;

    const bool maximizing = board.side == rootSide;
    const bool inCheck = isKingAttacked(board, board.side);
    orderMoves(board, legal, rootSide, ply);

    if (maximizing) {
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
            }
            auto delta = nnue_.computeMoveDelta(board, move);
            UndoInfo undo;
            applyMove(board, move, undo);
            if (ply + 1 < AccStackSize) nnue_.updateAccumulatorIncremental(accStack[ply], delta, accStack[ply + 1]);
            best = std::max(best, quiescence(board, depth - 1, ply + 1, alpha, beta, rootSide));
            undoMove(board, move, undo);
            alpha = std::max(alpha, best);
            if (alpha >= beta) break;
        }
        return best;
    }

    if (standPat <= alpha) return standPat;
    if (!inCheck && standPat - deltaMargin_ >= beta) return standPat;
    beta = std::min(beta, standPat);
    int best = standPat;
    for (const Move& move : legal) {
        if (!inCheck) {
            const bool capture = !move.isDrop() && board.squares[move.to] != 0;
            const bool promotion = move.promote;
            if (!capture && !promotion) {
                if (depth < qCheckDepthMin_ || !givesCheck(board, move)) continue;
            }
        }
        auto delta = nnue_.computeMoveDelta(board, move);
        UndoInfo undo;
        applyMove(board, move, undo);
        if (ply + 1 < AccStackSize) nnue_.updateAccumulatorIncremental(accStack[ply], delta, accStack[ply + 1]);
        best = std::min(best, quiescence(board, depth - 1, ply + 1, alpha, beta, rootSide));
        undoMove(board, move, undo);
        beta = std::min(beta, best);
        if (alpha >= beta) break;
    }
    return best;
}

void NNUEEngine::orderMoves(const Board& board, MoveList& moves, Color rootSide, int ply, const Move& prevMove) const {
    const std::uint64_t key = boardHash(board, rootSide);
    const int ttIndex = static_cast<int>(key & TTMask);
    const int lockIndex = static_cast<int>((key >> TTBits) % LockCount);
    Move ttMove{};
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.key == key && slot.generation == ttGeneration_) ttMove = slot.bestMove;
    }
    struct ScoredMove { Move move; int score; };
    const int n = moves.size();
    ScoredMove scored[MoveList::Capacity];
    for (int i = 0; i < n; ++i) scored[i] = {moves[i], moveOrderScore(board, moves[i], rootSide, ply, ttMove, prevMove)};
    std::stable_sort(scored, scored + n, [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });
    for (int i = 0; i < n; ++i) moves[i] = scored[i].move;
}

int NNUEEngine::moveOrderScore(const Board& board, const Move& move, Color rootSide, int ply, const Move& ttMove, const Move& prevMove) const {
    (void)rootSide;
    if (ttMove.to >= 0 && sameMove(move, ttMove)) return 20000;

    int score = 0;
    if (!move.isDrop() && board.squares[move.to] != 0) {
        const PieceType captured = typeOf(board.squares[move.to]);
        const PieceType moving = typeOf(board.squares[move.from]);
        score += 10000 + pieceValue(captured) * 10 - pieceValue(moving);
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
}

bool NNUEEngine::shouldStop() const {
    if (stopped_.load()) return true;
    if (std::chrono::steady_clock::now() >= deadline_) { stopped_.store(true); return true; }
    return false;
}

std::uint64_t NNUEEngine::boardHash(const Board& board, Color rootSide) const {
    return board.hash ^ (rootSide == Black ? 0x9E3779B97F4A7C15ULL : 0x6C62272E07BB0142ULL);
}

void NNUEEngine::setLastSearchInfo(const SearchInfo& info) const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    lastSearchInfo_ = info;
}

SearchInfo NNUEEngine::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    return lastSearchInfo_;
}

std::vector<Move> NNUEEngine::extractPV(Board board, Color rootSide, int maxDepth) const {
    std::vector<Move> pv;
    for (int i = 0; i < maxDepth; ++i) {
        const std::uint64_t key = boardHash(board, rootSide);
        const int ttIndex = static_cast<int>(key & TTMask);
        const int lockIndex = static_cast<int>((key >> TTBits) % LockCount);
        Move ttMove;
        {
            std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
            const TranspositionEntry& slot = transposition_[ttIndex];
            if (slot.key != key || slot.generation != ttGeneration_) break;
            ttMove = slot.bestMove;
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

void NNUEEngine::setSearchDepth(int depth) { searchDepth_ = std::clamp(depth, 0, 128); }
void NNUEEngine::setMaxMoveTimeMs(int ms) { maxMoveTimeMs_ = std::clamp(ms, 50, 600000); }
void NNUEEngine::setThreads(int threads) { threads_ = std::clamp(threads, 1, 256); }
int NNUEEngine::threadCount() const { return threads_; }
void NNUEEngine::setBookEnabled(bool enabled) { bookEnabled_ = enabled; }
bool NNUEEngine::loadBook(const std::string& path) { return book_.load(path); }
bool NNUEEngine::loadNNUE(const std::string& path) { return nnue_.load(path); }

void NNUEEngine::setParam(const std::string& name, int value) {
    if (name == "LMRFullDepthMoves") lmrFullDepthMoves_ = value;
    else if (name == "LMRMinDepth") lmrMinDepth_ = value;
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
    if (name == "LMRFullDepthMoves") return lmrFullDepthMoves_;
    if (name == "LMRMinDepth") return lmrMinDepth_;
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
