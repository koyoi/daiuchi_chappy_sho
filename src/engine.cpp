#include "engine.h"

#include "movegen.h"
#include "notation.h"
#include "position.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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
constexpr int QuiescenceDepth = 6;
constexpr int DeltaMargin = 1400;
constexpr int QCheckDepthMin = 4;
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
    case ProPawn:
    case ProLance:
    case ProKnight:
    case ProSilver:
        return 560;
    case Horse: return 1150;
    case Dragon: return 1350;
    default:
        return 0;
    }
}

int detectPhysicalCores() {
#ifdef _WIN32
    DWORD length = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return 0;
    }
    std::vector<unsigned char> buffer(length);
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
        return 0;
    }
    int cores = 0;
    DWORD offset = 0;
    while (offset < length) {
        auto* current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (current->Relationship == RelationProcessorCore) {
            ++cores;
        }
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

int openingSafetyScale(int moveNumber) {
    if (moveNumber >= 56) {
        return 0;
    }
    if (moveNumber <= 24) {
        return 100;
    }
    return std::max(0, (56 - moveNumber) * 100 / 32);
}

int promotionGain(PieceType type) {
    if (!canPromote(type)) {
        return 0;
    }
    return std::max(0, pieceValue(promote(type)) - pieceValue(type));
}

int vulnerablePiecePenalty(const Board& board, Color side) {
    int penalty = 0;
    const Color enemy = opposite(side);
    for (int square = 0; square < BoardSize; ++square) {
        const int piece = board.squares[square];
        if (piece == 0 || colorOf(piece) != side || typeOf(piece) == King) {
            continue;
        }
        const PieceType type = typeOf(piece);
        const int value = pieceValue(type);
        const int attackers = countAttackers(board, square, enemy);
        if (attackers <= 0) {
            continue;
        }
        const int defenders = countAttackers(board, square, side);
        if (defenders == 0) {
            penalty += value / 2;
            if (type == Bishop || type == Rook || type == Horse || type == Dragon) {
                penalty += value / 3;
            }
        } else if (attackers > defenders) {
            penalty += value / 4;
        }
    }
    return penalty;
}

int kingExposurePenalty(const Board& board, Color side) {
    const int king = findKing(board, side);
    if (king < 0) {
        return MateScore / 2;
    }
    const Color enemy = opposite(side);
    int penalty = 0;
    for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
            const int file = fileOf(king) + df;
            const int rank = rankOf(king) + dr;
            if (!inside(file, rank)) {
                continue;
            }
            const int square = idx(file, rank);
            const int attackers = countAttackers(board, square, enemy);
            if (attackers > 0) {
                penalty += df == 0 && dr == 0 ? 420 * attackers : 95 * attackers;
            }
        }
    }
    return penalty;
}

int opponentImmediateThreatScore(const Board& board, Color targetSide) {
    if (board.side == targetSide) {
        return 0;
    }
    const int king = findKing(board, targetSide);
    const auto replies = generateLegalMoves(board, true);
    Board temp = board;
    int best = 0;
    int accumulated = 0;
    int checks = 0;
    for (const Move& reply : replies) {
        int score = 0;
        if (!reply.isDrop() && board.squares[reply.to] != 0 && colorOf(board.squares[reply.to]) == targetSide) {
            const PieceType captured = typeOf(board.squares[reply.to]);
            score += pieceValue(captured);
            if (captured == Bishop || captured == Rook || captured == Horse || captured == Dragon) {
                score += 260;
            }
        }
        if (reply.promote) {
            score += 260 + promotionGain(reply.piece);
        }
        if (king >= 0) {
            const int distance = chebyshevDistance(reply.to, king);
            if (reply.isDrop() && distance <= 2) {
                score += pieceValue(reply.drop) / 2 + (distance <= 1 ? 180 : 70);
            } else if (distance <= 1 && !reply.isDrop()) {
                score += 80;
            }
        }

        UndoInfo undo;
        applyMove(temp, reply, undo);
        if (isKingAttacked(temp, targetSide)) {
            score += 1100;
            ++checks;
            if (generateLegalMoves(temp, true).empty()) {
                undoMove(temp, reply, undo);
                return MateScore;
            }
        }
        undoMove(temp, reply, undo);

        if (score > 0) {
            best = std::max(best, score);
            accumulated += std::min(score, 1400) / 5;
        }
    }
    return best + accumulated + checks * 160;
}

int openingTrapPenalty(const Board& before, const Move& move, Color side) {
    const int scale = openingSafetyScale(before.moveNumber);
    if (scale <= 0) {
        return 0;
    }
    Board next = before;
    applyMove(next, move);
    const int threat = opponentImmediateThreatScore(next, side);
    const int vulnerable = vulnerablePiecePenalty(next, side);
    const int kingExposure = kingExposurePenalty(next, side);
    return (threat + vulnerable + kingExposure) * scale / 100;
}

} // namespace

LearningEngine::LearningEngine()
    : learner_(evaluator_),
      transposition_(TTSize),
      rng_(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())) {
    zobrist::init();
    threads_ = defaultThreadCount();
    learner_.loadWeights();
}

Move LearningEngine::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move LearningEngine::chooseMove(const Board& board, const SearchLimits& limits) {
    return chooseMove(board, limits, InfoCallback{});
}

Move LearningEngine::chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback) {
    const auto searchStart = std::chrono::steady_clock::now();
    stopped_.store(false);
    nodes_.store(0);
    ++ttGeneration_;
    const int requestedMoveTime = limits.moveTimeMs > 0 ? limits.moveTimeMs : maxMoveTimeMs_;
    const int moveTime = std::clamp(requestedMoveTime, 50, 600000);
    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(moveTime);

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        SearchInfo info;
        info.depth = 0;
        info.nodes = nodes_.load();
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
                if (sameMove(m, bookMove)) {
                    bookMove = m;
                    isLegal = true;
                    break;
                }
            }
            if (isLegal) {
                SearchInfo info;
                info.depth = 0;
                info.scoreCp = 0;
                info.nodes = 0;
                info.timeMs = 0;
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
        const int mateBudget = std::min(moveTime / 10, 200);
        MateResult mateResult = mateSolver_.searchMate(board, rootSide, 7, mateBudget);
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

    analyzeSubGoals(board, rootSide);

    orderMoves(board, legal, rootSide, 0);

    std::vector<int> openingPenalties(legal.size(), 0);
    if (openingSafety_) {
        for (int i = 0; i < legal.size(); ++i) {
            openingPenalties[i] = openingTrapPenalty(board, legal[i], rootSide);
        }
    }

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
            aspirationAlpha = prevIterScore - AspirationWindow;
            aspirationBeta = prevIterScore + AspirationWindow;
        }

        std::vector<int> scores = scoreRootMovesParallel(board, legal, openingPenalties, rootSide, depth, pruneWidth, aspirationAlpha, aspirationBeta, searchStart, infoCallback);

        if (!shouldStop() && depth >= 2 && std::abs(prevIterScore) < MateScore / 2) {
            int depthBest = std::numeric_limits<int>::min();
            for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
                if (scores[i] != std::numeric_limits<int>::min()) {
                    depthBest = std::max(depthBest, scores[i]);
                }
            }
            if (depthBest != std::numeric_limits<int>::min() && (depthBest <= aspirationAlpha || depthBest >= aspirationBeta)) {
                scores = scoreRootMovesParallel(board, legal, openingPenalties, rootSide, depth, pruneWidth, -MateScore, MateScore, searchStart, infoCallback);
            }
        }

        if (shouldStop()) {
            break;
        }

        int depthBestScore = std::numeric_limits<int>::min();
        std::vector<Move> depthBestMoves;
        for (int i = 0; i < legal.size(); ++i) {
            const int score = scores[i];
            if (score == std::numeric_limits<int>::min()) {
                continue;
            }
            if (score > depthBestScore) {
                depthBestScore = score;
                depthBestMoves.clear();
                depthBestMoves.push_back(legal[i]);
            } else if (score == depthBestScore) {
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
            if (infoCallback) {
                infoCallback(info);
            }

            // Reorder root moves by score for next iteration (best first)
            struct IndexedScore { int index; int score; };
            std::vector<IndexedScore> ranked(legal.size());
            for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
                ranked[i] = {i, scores[i]};
            }
            std::stable_sort(ranked.begin(), ranked.end(), [](const IndexedScore& a, const IndexedScore& b) {
                return a.score > b.score;
            });
            MoveList reordered;
            std::vector<int> reorderedPenalties(openingPenalties.size(), 0);
            for (int i = 0; i < static_cast<int>(ranked.size()); ++i) {
                reordered.push_back(legal[ranked[i].index]);
                if (ranked[i].index < static_cast<int>(openingPenalties.size())) {
                    reorderedPenalties[i] = openingPenalties[ranked[i].index];
                }
            }
            legal = reordered;
            openingPenalties = reorderedPenalties;
            prevIterScore = depthBestScore;
        }
    }

    if (bestMoves.empty()) {
        bestMoves.push_back(legal.front());
    }
    std::uniform_int_distribution<std::size_t> dist(0, bestMoves.size() - 1);
    Move selected = bestMoves[dist(rng_)];
    // Final safety check: ensure selected move is legal and does not leave our king in check.
    {
        bool legalFound = false;
        for (const Move& m : legal) {
            if (sameMove(m, selected)) { legalFound = true; break; }
        }
        if (legalFound) {
            Board tmp = board;
            applyMove(tmp, selected);
            if (isKingAttacked(tmp, opposite(tmp.side))) {
                legalFound = false;
            }
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

void LearningEngine::recordMove(const Board& before, const Move& move, bool engineMove) {
    learner_.recordMove(before, move, engineMove);
}

void LearningEngine::finishGame(int engineResult, Color engineSide) {
    learner_.finishGame(engineResult, engineSide);
}

void LearningEngine::clearGame() {
    learner_.clearGame();
}

void LearningEngine::setLearningEnabled(bool enabled) {
    learner_.setEnabled(enabled);
}

void LearningEngine::setRecordOnly(bool recordOnly) {
    learner_.setRecordOnly(recordOnly);
}

void LearningEngine::setSearchDepth(int depth) {
    searchDepth_ = std::clamp(depth, 0, 128);
}

void LearningEngine::setMaxMoveTimeMs(int milliseconds) {
    maxMoveTimeMs_ = std::clamp(milliseconds, 50, 600000);
}

void LearningEngine::setHeavyEvaluation(bool enabled) {
    evaluator_.setHeavyFeatures(enabled);
}

void LearningEngine::setOpeningSafety(bool enabled) {
    openingSafety_ = enabled;
}

void LearningEngine::setThreads(int threads) {
    threads_ = std::clamp(threads, 1, 256);
}

int LearningEngine::threadCount() const {
    return threads_;
}

void LearningEngine::setWeightsPath(const std::string& path) {
    learner_.setWeightsPath(path);
}

void LearningEngine::setTrainingDataPath(const std::string& path) {
    learner_.setTrainingDataPath(path);
}

bool LearningEngine::loadWeights() {
    return learner_.loadWeights();
}

const std::string& LearningEngine::weightsPath() const {
    return learner_.weightsPath();
}

bool LearningEngine::loadMlpWeights(const std::string& path) {
    return evaluator_.loadMlp(path);
}

void LearningEngine::setRootPruneWidth(int width) {
    rootPruneWidth_ = std::max(0, width);
}

void LearningEngine::setBookEnabled(bool enabled) {
    bookEnabled_ = enabled;
}

bool LearningEngine::loadBook(const std::string& path) {
    return book_.load(path);
}

SearchInfo LearningEngine::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    return lastSearchInfo_;
}

void LearningEngine::clearSearchTables() const {
    for (auto& ply : killers_) {
        for (auto& slot : ply) {
            slot = Move{};
        }
    }
    for (auto& color : history_) {
        for (auto& from : color) {
            for (auto& val : from) {
                val /= 2;
            }
        }
    }
    for (auto& color : counterMoves_) {
        for (auto& from : color) {
            for (auto& m : from) {
                m = Move{};
            }
        }
    }
}

void LearningEngine::storeCounterMove(Color side, const Move& prevMove, const Move& counterMove) const {
    if (prevMove.to < 0) return;
    const int colorIdx = side == Black ? 0 : 1;
    const int from = prevMove.isDrop() ? prevMove.to : prevMove.from;
    if (from < 0 || from >= BoardSize || prevMove.to < 0 || prevMove.to >= BoardSize) return;
    counterMoves_[colorIdx][from][prevMove.to] = counterMove;
}

void LearningEngine::analyzeSubGoals(const Board& board, Color rootSide) const {
    std::fill(targetPieces_.begin(), targetPieces_.end(), false);
    const int baseScore = evaluator_.evaluate(board, rootSide);
    const int ci = rootSide == Black ? 0 : 1;

    for (PieceType pt : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        if (hand(board, rootSide)[pt] > 0) continue;

        Board hyp = board;
        const int oldCount = hand(hyp, rootSide)[pt];
        hyp.hash ^= zobrist::handKey(ci, pt, oldCount);
        hand(hyp, rootSide)[pt]++;
        hyp.hash ^= zobrist::handKey(ci, pt, oldCount + 1);

        const int hypScore = search(hyp, SubGoalDepth, 0, -MateScore, MateScore, rootSide, false, Move{});
        if (hypScore - baseScore >= SubGoalThreshold) {
            targetPieces_[pt] = true;
        }
    }
}

void LearningEngine::storeKiller(int ply, const Move& move) const {
    if (ply >= MaxPly) return;
    if (sameMove(move, killers_[ply][0])) return;
    killers_[ply][1] = killers_[ply][0];
    killers_[ply][0] = move;
}

void LearningEngine::updateHistory(Color side, const Move& move, int depth, bool good) const {
    if (move.isDrop() || move.promote) return;
    if (move.from < 0 || move.from >= BoardSize || move.to < 0 || move.to >= BoardSize) return;
    const int colorIdx = side == Black ? 0 : 1;
    const int bonus = good ? depth * depth : -(depth * depth);
    int entry = history_[colorIdx][move.from][move.to];
    entry += bonus - entry * std::abs(bonus) / 16384;
    history_[colorIdx][move.from][move.to] = static_cast<std::int16_t>(std::clamp(entry, -16384, 16384));
}

int LearningEngine::search(Board& board, int depth, int ply, int alpha, int beta, Color rootSide, bool allowNullMove, const Move& prevMove) const {
    nodes_.fetch_add(1);
    if (shouldStop()) {
        return evaluator_.evaluate(board, rootSide);
    }

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
                if (slot.flag == ExactScore) {
                    return slot.score;
                }
                if (slot.flag == LowerBound) {
                    alpha = std::max(alpha, slot.score);
                } else if (slot.flag == UpperBound) {
                    beta = std::min(beta, slot.score);
                }
                if (alpha >= beta) {
                    return slot.score;
                }
            }
        }
    }

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side)) {
            return board.side == rootSide ? -MateScore - depth : MateScore + depth;
        }
        return 0;
    }

    if (depth <= 0) {
        return quiescence(board, QuiescenceDepth, ply, alpha, beta, rootSide);
    }

    const bool inCheck = isKingAttacked(board, board.side);
    const bool maximizing = board.side == rootSide;

    // Null Move Pruning
    if (allowNullMove && !inCheck && depth >= NMPMinDepth && ply > 0) {
        NullMoveUndoInfo nullUndo;
        applyNullMove(board, nullUndo);
        const int nullVal = search(board, depth - 1 - NMPReduction, ply + 1, alpha, beta, rootSide, false, Move{});
        undoNullMove(board, nullUndo);

        if (maximizing) {
            if (nullVal >= beta) {
                return nullVal;
            }
        } else {
            if (nullVal <= alpha) {
                return nullVal;
            }
        }
    }

    // Internal Iterative Deepening
    if (ttMove.to < 0 && depth >= IIDMinDepth && !inCheck) {
        search(board, depth - 3, ply, alpha, beta, rootSide, false, prevMove);
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.key == key && slot.generation == ttGeneration_) {
            ttMove = slot.bestMove;
        }
    }

    orderMoves(board, legal, rootSide, ply, prevMove);

    // Futility pruning setup
    const bool canFutilityPrune = !inCheck && (depth == 1 || depth == 2);
    int staticEval = 0;
    if (canFutilityPrune) {
        staticEval = evaluator_.evaluate(board, rootSide);
    }
    const int futilityMargin = depth == 1 ? FutilityMargin1 : FutilityMargin2;

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

            if (canFutilityPrune && isQuiet && !isCheck && staticEval + futilityMargin <= alpha) {
                ++moveIndex;
                continue;
            }

            UndoInfo undo;
            applyMove(board, move, undo);

            const bool givesCheckNow = isKingAttacked(board, board.side);
            int extension = (givesCheckNow && ply < MaxPly - 10) ? 1 : 0;
            const int newDepth = depth - 1 + extension;

            int val = 0;
            if (moveIndex == 0) {
                val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
            } else {
                bool needsFullWindow = false;
                if (depth >= LMRMinDepth && moveIndex >= LMRFullDepthMoves && isQuiet && !isCheck && !inCheck && !givesCheckNow) {
                    int R = 1 + (depth >= 6 ? 1 : 0) + (moveIndex >= 10 ? 1 : 0);
                    int reducedDepth = std::max(1, newDepth - R);
                    val = search(board, reducedDepth, ply + 1, alpha, alpha + 1, rootSide, true, move);
                    if (val > alpha) {
                        val = search(board, newDepth, ply + 1, alpha, alpha + 1, rootSide, true, move);
                        if (val > alpha && val < beta) {
                            needsFullWindow = true;
                        }
                    }
                } else {
                    val = search(board, newDepth, ply + 1, alpha, alpha + 1, rootSide, true, move);
                    if (val > alpha && val < beta) {
                        needsFullWindow = true;
                    }
                }
                if (needsFullWindow) {
                    val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
                }
            }

            undoMove(board, move, undo);
            anySearched = true;

            if (val > best) {
                best = val;
                bestMoveLocal = move;
            }
            alpha = std::max(alpha, best);
            if (alpha >= beta) {
                if (isQuiet) {
                    storeKiller(ply, move);
                    updateHistory(board.side, move, depth, true);
                    storeCounterMove(board.side, prevMove, move);
                    for (int q = 0; q < quietsCount; ++q) {
                        updateHistory(board.side, quietsTried[q], depth, false);
                    }
                }
                break;
            }
            if (isQuiet && quietsCount < MoveList::Capacity) {
                quietsTried[quietsCount++] = move;
            }
            ++moveIndex;
        }

        if (!anySearched) {
            return canFutilityPrune ? staticEval : (board.side == rootSide ? -MateScore - depth : MateScore + depth);
        }

        {
            std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
            TranspositionEntry& slot = transposition_[ttIndex];
            if (slot.generation != ttGeneration_ || depth >= slot.depth) {
                slot.key = key;
                slot.depth = depth;
                slot.score = best;
                slot.bestMove = bestMoveLocal;
                slot.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
                slot.generation = ttGeneration_;
            }
        }
        return best;
    }

    // Minimizing branch
    int best = std::numeric_limits<int>::max();
    Move bestMoveLocal{};
    Move quietsTried[MoveList::Capacity];
    int quietsCount = 0;
    int moveIndex = 0;
    bool anySearched = false;

    for (const Move& move : legal) {
        const bool isQuiet = (move.isDrop() || board.squares[move.to] == 0) && !move.promote;
        const bool isCheck = givesCheck(board, move);

        if (canFutilityPrune && isQuiet && !isCheck && staticEval - futilityMargin >= beta) {
            ++moveIndex;
            continue;
        }

        UndoInfo undo;
        applyMove(board, move, undo);

        const bool givesCheckNow = isKingAttacked(board, board.side);
        int extension = (givesCheckNow && ply < MaxPly - 10) ? 1 : 0;
        const int newDepth = depth - 1 + extension;

        int val = 0;
        if (moveIndex == 0) {
            val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
        } else {
            bool needsFullWindow = false;
            if (depth >= LMRMinDepth && moveIndex >= LMRFullDepthMoves && isQuiet && !isCheck && !inCheck && !givesCheckNow) {
                int R = 1 + (depth >= 6 ? 1 : 0) + (moveIndex >= 10 ? 1 : 0);
                int reducedDepth = std::max(1, newDepth - R);
                val = search(board, reducedDepth, ply + 1, beta - 1, beta, rootSide, true, move);
                if (val < beta) {
                    val = search(board, newDepth, ply + 1, beta - 1, beta, rootSide, true, move);
                    if (val < beta && val > alpha) {
                        needsFullWindow = true;
                    }
                }
            } else {
                val = search(board, newDepth, ply + 1, beta - 1, beta, rootSide, true, move);
                if (val < beta && val > alpha) {
                    needsFullWindow = true;
                }
            }
            if (needsFullWindow) {
                val = search(board, newDepth, ply + 1, alpha, beta, rootSide, true, move);
            }
        }

        undoMove(board, move, undo);
        anySearched = true;

        if (val < best) {
            best = val;
            bestMoveLocal = move;
        }
        beta = std::min(beta, best);
        if (alpha >= beta) {
            if (isQuiet) {
                storeKiller(ply, move);
                updateHistory(board.side, move, depth, true);
                storeCounterMove(board.side, prevMove, move);
                for (int q = 0; q < quietsCount; ++q) {
                    updateHistory(board.side, quietsTried[q], depth, false);
                }
            }
            break;
        }
        if (isQuiet && quietsCount < MoveList::Capacity) {
            quietsTried[quietsCount++] = move;
        }
        ++moveIndex;
    }

    if (!anySearched) {
        return canFutilityPrune ? staticEval : (board.side == rootSide ? -MateScore - depth : MateScore + depth);
    }

    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.generation != ttGeneration_ || depth >= slot.depth) {
            slot.key = key;
            slot.depth = depth;
            slot.score = best;
            slot.bestMove = bestMoveLocal;
            slot.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
            slot.generation = ttGeneration_;
        }
    }
    return best;
}

int LearningEngine::quiescence(Board& board, int depth, int ply, int alpha, int beta, Color rootSide) const {
    nodes_.fetch_add(1);
    if (shouldStop()) {
        return evaluator_.evaluate(board, rootSide);
    }

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side)) {
            return board.side == rootSide ? -MateScore - depth : MateScore + depth;
        }
        return 0;
    }

    int standPat = evaluator_.evaluate(board, rootSide);
    if (depth <= 0) {
        return standPat;
    }

    const bool maximizing = board.side == rootSide;
    const bool inCheck = isKingAttacked(board, board.side);

    orderMoves(board, legal, rootSide, ply);

    if (maximizing) {
        if (standPat >= beta) {
            return standPat;
        }
        if (!inCheck && standPat + DeltaMargin <= alpha) {
            return standPat;
        }
        alpha = std::max(alpha, standPat);
        int best = standPat;
        for (const Move& move : legal) {
            if (!inCheck) {
                const bool capture = !move.isDrop() && board.squares[move.to] != 0;
                const bool promotion = move.promote;
                if (!capture && !promotion) {
                    if (depth < QCheckDepthMin || !givesCheck(board, move)) {
                        continue;
                    }
                }
            }
            UndoInfo undo;
            applyMove(board, move, undo);
            best = std::max(best, quiescence(board, depth - 1, ply + 1, alpha, beta, rootSide));
            undoMove(board, move, undo);
            alpha = std::max(alpha, best);
            if (alpha >= beta) {
                break;
            }
        }
        return best;
    }

    if (standPat <= alpha) {
        return standPat;
    }
    if (!inCheck && standPat - DeltaMargin >= beta) {
        return standPat;
    }
    beta = std::min(beta, standPat);
    int best = standPat;
    for (const Move& move : legal) {
        if (!inCheck) {
            const bool capture = !move.isDrop() && board.squares[move.to] != 0;
            const bool promotion = move.promote;
            if (!capture && !promotion) {
                if (depth < QCheckDepthMin || !givesCheck(board, move)) {
                    continue;
                }
            }
        }
        UndoInfo undo;
        applyMove(board, move, undo);
        best = std::min(best, quiescence(board, depth - 1, ply + 1, alpha, beta, rootSide));
        undoMove(board, move, undo);
        beta = std::min(beta, best);
        if (alpha >= beta) {
            break;
        }
    }
    return best;
}

bool LearningEngine::canForceMate(Board& board, int depth, Color attacker) const {
    nodes_.fetch_add(1);
    if (shouldStop()) {
        return false;
    }
    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        return isKingAttacked(board, board.side) && board.side != attacker;
    }
    if (depth <= 0) {
        return false;
    }

    orderMoves(board, legal, attacker, 0);

    if (board.side == attacker) {
        for (const Move& move : legal) {
            UndoInfo undo;
            applyMove(board, move, undo);
            const bool mate = canForceMate(board, depth - 1, attacker);
            undoMove(board, move, undo);
            if (mate) {
                return true;
            }
        }
        return false;
    }

    for (const Move& move : legal) {
        UndoInfo undo;
        applyMove(board, move, undo);
        const bool mate = canForceMate(board, depth - 1, attacker);
        undoMove(board, move, undo);
        if (!mate) {
            return false;
        }
    }
    return true;
}

bool LearningEngine::isTacticalMove(const Board& board, const Move& move) const {
    if (!move.isDrop() && board.squares[move.to] != 0) {
        return true;
    }
    if (move.promote) {
        return true;
    }
    return givesCheck(board, move);
}

std::vector<int> LearningEngine::scoreRootMovesParallel(
    const Board& board,
    const MoveList& orderedMoves,
    const std::vector<int>& openingPenalties,
    Color rootSide,
    int depth,
    int pruneWidth,
    int aspirationAlpha,
    int aspirationBeta,
    const std::chrono::steady_clock::time_point& searchStart,
    const InfoCallback& infoCallback) const {
    std::vector<int> scores(orderedMoves.size(), std::numeric_limits<int>::min());
    if (orderedMoves.empty()) {
        return scores;
    }

    std::mutex bestMutex;
    std::mutex emitMutex;
    int bestScore = std::numeric_limits<int>::min();
    Move bestMove;
    bool hasBestMove = false;
    auto lastEmit = searchStart - std::chrono::milliseconds(1000);
    std::atomic<int> sharedAlpha{aspirationAlpha};

    auto publish = [&](int index, int score, bool force) {
        SearchInfo info;
        bool shouldEmit = false;
        Move currentBest;
        {
            std::lock_guard<std::mutex> lock(bestMutex);
            if (!hasBestMove || score > bestScore) {
                bestScore = score;
                bestMove = orderedMoves[index];
                hasBestMove = true;
            }
            currentBest = bestMove;
            const auto now = std::chrono::steady_clock::now();
            if (force || now - lastEmit >= std::chrono::milliseconds(250)) {
                lastEmit = now;
                info.depth = depth;
                info.scoreCp = std::clamp(bestScore, -MateScore, MateScore);
                info.nodes = nodes_.load();
                info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - searchStart).count());
                info.bestMove = currentBest;
                info.hasBestMove = hasBestMove;
                shouldEmit = hasBestMove;
            }
        }
        if (shouldEmit && infoCallback) {
            info.pv.push_back(currentBest);
            Board pvBoard = board;
            applyMove(pvBoard, currentBest);
            auto pvTail = extractPV(pvBoard, rootSide, depth);
            info.pv.insert(info.pv.end(), pvTail.begin(), pvTail.end());
            std::lock_guard<std::mutex> emitLock(emitMutex);
            infoCallback(info);
        }
    };

    const int workerCount = std::min<int>(std::max(1, threads_), orderedMoves.size());
    if (workerCount <= 1) {
        int runningAlpha = aspirationAlpha;
        for (int i = 0; i < orderedMoves.size(); ++i) {
            if (shouldStop()) {
                break;
            }
            Board next = board;
            applyMove(next, orderedMoves[i]);
            int searchDepth = depth - 1;
            if (pruneWidth > 0 && depth >= 3 && i >= pruneWidth) {
                searchDepth = std::max(0, depth - 3);
            }
            scores[i] = search(next, searchDepth, 1, runningAlpha, aspirationBeta, rootSide, true, orderedMoves[i]);
            if (i < static_cast<int>(openingPenalties.size())) {
                scores[i] -= openingPenalties[i];
            }
            runningAlpha = std::max(runningAlpha, scores[i]);
            publish(i, scores[i], i + 1 == orderedMoves.size());
        }
        return scores;
    }

    std::atomic_int nextIndex{0};
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (int worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&, rootSide, pruneWidth, aspirationBeta]() {
            while (!shouldStop()) {
                const int index = nextIndex.fetch_add(1);
                if (index >= static_cast<int>(orderedMoves.size())) {
                    break;
                }
                Board next = board;
                applyMove(next, orderedMoves[index]);
                int searchDepth = depth - 1;
                if (pruneWidth > 0 && depth >= 3 && index >= pruneWidth) {
                    searchDepth = std::max(0, depth - 3);
                }
                const int localAlpha = sharedAlpha.load(std::memory_order_relaxed);
                scores[index] = search(next, searchDepth, 1, localAlpha, aspirationBeta, rootSide, true, orderedMoves[index]);
                if (index < static_cast<int>(openingPenalties.size())) {
                    scores[index] -= openingPenalties[index];
                }
                int expected = sharedAlpha.load(std::memory_order_relaxed);
                while (scores[index] > expected) {
                    if (sharedAlpha.compare_exchange_weak(expected, scores[index], std::memory_order_relaxed)) {
                        break;
                    }
                }
                publish(index, scores[index], index + 1 == static_cast<int>(orderedMoves.size()));
            }
        });
    }
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    return scores;
}

void LearningEngine::orderMoves(const Board& board, MoveList& moves, Color rootSide, int ply, const Move& prevMove) const {
    const std::uint64_t key = boardHash(board, rootSide);
    const int ttIndex = static_cast<int>(key & TTMask);
    const int lockIndex = static_cast<int>((key >> TTBits) % LockCount);
    Move ttMove{};
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.key == key && slot.generation == ttGeneration_) {
            ttMove = slot.bestMove;
        }
    }

    struct ScoredMove {
        Move move;
        int score;
    };
    const int n = moves.size();
    ScoredMove scored[MoveList::Capacity];
    for (int i = 0; i < n; ++i) {
        scored[i] = {moves[i], moveOrderScore(board, moves[i], rootSide, ply, ttMove, prevMove)};
    }
    std::stable_sort(scored, scored + n, [](const ScoredMove& left, const ScoredMove& right) {
        return left.score > right.score;
    });
    for (int i = 0; i < n; ++i) {
        moves[i] = scored[i].move;
    }
}

int LearningEngine::moveOrderScore(const Board& board, const Move& move, Color rootSide, int ply, const Move& ttMove, const Move& prevMove) const {
    (void)rootSide;

    if (ttMove.to >= 0 && sameMove(move, ttMove)) {
        return 20000;
    }

    int score = 0;

    if (!move.isDrop() && board.squares[move.to] != 0) {
        const PieceType captured = typeOf(board.squares[move.to]);
        const PieceType moving = typeOf(board.squares[move.from]);
        score += 10000 + pieceValue(captured) * 10 - pieceValue(moving);
        const PieceType capturedBase = unpromote(captured);
        if (targetPieces_[capturedBase]) {
            score += SubGoalCaptureBonus;
        }
    }

    if (givesCheck(board, move)) {
        score += 8000;
    }

    if (move.promote) {
        score += 6000;
    }

    if (score >= 6000) {
        return score;
    }

    if (ply < MaxPly) {
        if (sameMove(move, killers_[ply][0])) {
            return 5000;
        }
        if (sameMove(move, killers_[ply][1])) {
            return 4900;
        }
    }

    if (prevMove.to >= 0) {
        const int colorIdx = board.side == Black ? 0 : 1;
        const int prevFrom = prevMove.isDrop() ? prevMove.to : prevMove.from;
        if (prevFrom >= 0 && prevFrom < BoardSize && prevMove.to >= 0 && prevMove.to < BoardSize) {
            if (sameMove(move, counterMoves_[colorIdx][prevFrom][prevMove.to])) {
                return 4800;
            }
        }
    }

    if (move.isDrop()) {
        score += pieceValue(move.drop) / 3;

        const int enemyKing = board.side == Black ? board.whiteKingSquare : board.blackKingSquare;
        if (enemyKing >= 0 && chebyshevDistance(move.to, enemyKing) <= 2) {
            score += 200;
        }
    } else if (board.squares[move.to] == 0) {
        const int colorIdx = board.side == Black ? 0 : 1;
        if (move.from >= 0 && move.from < BoardSize) {
            score += history_[colorIdx][move.from][move.to];
        }

        const Color enemy = opposite(board.side);
        if (isSquareAttacked(board, move.to, enemy)) {
            const int enemyKing = board.side == Black ? board.whiteKingSquare : board.blackKingSquare;
            if (enemyKing >= 0 && chebyshevDistance(move.to, enemyKing) <= 2) {
                score += 300;
            }
        }
    }

    return score;
}

bool LearningEngine::shouldStop() const {
    if (stopped_.load()) {
        return true;
    }
    if (std::chrono::steady_clock::now() >= deadline_) {
        stopped_.store(true);
        return true;
    }
    return false;
}

std::uint64_t LearningEngine::boardHash(const Board& board, Color rootSide) const {
    return board.hash ^ (rootSide == Black ? 0x9E3779B97F4A7C15ULL : 0x6C62272E07BB0142ULL);
}

void LearningEngine::setLastSearchInfo(const SearchInfo& info) const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    lastSearchInfo_ = info;
}

MateResult LearningEngine::searchMate(const Board& board, int timeLimitMs) {
    return mateSolver_.searchMate(board, board.side, 31, timeLimitMs);
}

std::vector<Move> LearningEngine::extractPV(Board board, Color rootSide, int maxDepth) const {
    std::vector<Move> pv;
    for (int i = 0; i < maxDepth; ++i) {
        const std::uint64_t key = boardHash(board, rootSide);
        const int ttIndex = static_cast<int>(key & TTMask);
        const int lockIndex = static_cast<int>((key >> TTBits) % LockCount);
        Move ttMove;
        {
            std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
            const TranspositionEntry& slot = transposition_[ttIndex];
            if (slot.key != key || slot.generation != ttGeneration_) {
                break;
            }
            ttMove = slot.bestMove;
        }
        if (ttMove.to < 0) {
            break;
        }
        pv.push_back(ttMove);
        applyMove(board, ttMove);
    }
    return pv;
}

int LearningEngine::depthLimit() const {
    return searchDepth_ > 0 ? searchDepth_ : 128;
}

} // namespace shogi
