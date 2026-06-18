#include "engine.h"

#include "movegen.h"
#include "position.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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

constexpr int MateScore = 29000;
constexpr int MateSearchDepth = 3;
constexpr int QuiescenceDepth = 4;
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

} // namespace

LearningEngine::LearningEngine()
    : learner_(evaluator_),
      rng_(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())) {
    threads_ = defaultThreadCount();
    learner_.loadWeights();
}

Move LearningEngine::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move LearningEngine::chooseMove(const Board& board, const SearchLimits& limits) {
    stopped_.store(false);
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_);
        transposition_.clear();
    }
    const int requestedMoveTime = limits.moveTimeMs > 0 ? limits.moveTimeMs : maxMoveTimeMs_;
    const int moveTime = std::clamp(requestedMoveTime, 50, 600000);
    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(moveTime);

    const auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        return Move{};
    }

    Move gpuMove;
    if (chooseMoveByGpu(board, legal, gpuMove)) {
        return gpuMove;
    }

    const Color rootSide = board.side;
    std::vector<Move> mateMoves;
    for (const Move& move : orderMoves(board, legal, rootSide)) {
        if (shouldStop()) {
            break;
        }
        Board next = board;
        applyMove(next, move);
        if (canForceMate(next, MateSearchDepth - 1, rootSide)) {
            mateMoves.push_back(move);
        }
    }
    if (!mateMoves.empty()) {
        std::uniform_int_distribution<std::size_t> dist(0, mateMoves.size() - 1);
        return mateMoves[dist(rng_)];
    }

    int bestScore = std::numeric_limits<int>::min();
    std::vector<Move> bestMoves;
    const std::vector<Move> orderedMoves = orderMoves(board, legal, rootSide);
    const std::vector<int> scores = scoreRootMovesParallel(board, orderedMoves, rootSide);
    for (std::size_t i = 0; i < orderedMoves.size(); ++i) {
        const int score = scores[i];
        if (score == std::numeric_limits<int>::min()) {
            continue;
        }
        if (score > bestScore) {
            bestScore = score;
            bestMoves.clear();
            bestMoves.push_back(orderedMoves[i]);
        } else if (score == bestScore) {
            bestMoves.push_back(orderedMoves[i]);
        }
    }

    if (bestMoves.empty()) {
        bestMoves.push_back(legal.front());
    }
    std::uniform_int_distribution<std::size_t> dist(0, bestMoves.size() - 1);
    return bestMoves[dist(rng_)];
}

void LearningEngine::recordMove(const Board& before, const Move& move, bool engineMove) {
    learner_.recordMove(before, move, engineMove);
}

void LearningEngine::finishGame(int engineResult, Color engineSide) {
    learner_.finishGame(engineResult, engineSide);
    gpu_.train(learner_.trainingDataPath());
}

void LearningEngine::clearGame() {
    learner_.clearGame();
}

void LearningEngine::setLearningEnabled(bool enabled) {
    learner_.setEnabled(enabled);
}

void LearningEngine::setSearchDepth(int depth) {
    searchDepth_ = std::clamp(depth, 1, 6);
}

void LearningEngine::setMaxMoveTimeMs(int milliseconds) {
    maxMoveTimeMs_ = std::clamp(milliseconds, 50, 600000);
}

void LearningEngine::setHeavyEvaluation(bool enabled) {
    evaluator_.setHeavyFeatures(enabled);
}

void LearningEngine::setThreads(int threads) {
    threads_ = std::clamp(threads, 1, 256);
}

int LearningEngine::threadCount() const {
    return threads_;
}

void LearningEngine::setWeightsPath(const std::string& path) {
    learner_.setWeightsPath(path);
    learner_.loadWeights();
}

void LearningEngine::setTrainingDataPath(const std::string& path) {
    learner_.setTrainingDataPath(path);
}

void LearningEngine::setGpuEnabled(bool enabled) {
    gpu_.setEnabled(enabled);
}

void LearningEngine::setGpuTrainOnGameEnd(bool enabled) {
    gpu_.setTrainOnGameEnd(enabled);
}

void LearningEngine::setGpuPython(const std::string& python) {
    gpu_.setPython(python);
}

void LearningEngine::setGpuScript(const std::string& script) {
    gpu_.setScript(script);
}

void LearningEngine::setGpuModel(const std::string& model) {
    gpu_.setModel(model);
}

void LearningEngine::setGpuDevice(const std::string& device) {
    gpu_.setDevice(device);
}

void LearningEngine::loadWeights() {
    learner_.loadWeights();
}

bool LearningEngine::chooseMoveByGpu(const Board& board, const std::vector<Move>& legal, Move& selected) {
    std::vector<FeatureVector> features;
    features.reserve(legal.size());
    for (const Move& move : legal) {
        Board next = board;
        applyMove(next, move);
        features.push_back(evaluator_.extractFeatures(next, board.side));
    }

    std::vector<int> scores;
    if (!gpu_.score(features, scores)) {
        return false;
    }

    int bestScore = std::numeric_limits<int>::min();
    std::vector<Move> bestMoves;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        if (scores[i] > bestScore) {
            bestScore = scores[i];
            bestMoves.clear();
            bestMoves.push_back(legal[i]);
        } else if (scores[i] == bestScore) {
            bestMoves.push_back(legal[i]);
        }
    }
    if (bestMoves.empty()) {
        return false;
    }
    std::uniform_int_distribution<std::size_t> dist(0, bestMoves.size() - 1);
    selected = bestMoves[dist(rng_)];
    return true;
}

int LearningEngine::search(Board board, int depth, int alpha, int beta, Color rootSide) const {
    if (shouldStop()) {
        return evaluator_.evaluate(board, rootSide);
    }

    const int alphaOriginal = alpha;
    const int betaOriginal = beta;
    const std::uint64_t key = boardHash(board, rootSide);
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_);
        const auto hit = transposition_.find(key);
        if (hit != transposition_.end() && hit->second.depth >= depth) {
            const TranspositionEntry entry = hit->second;
            if (entry.flag == ExactScore) {
                return entry.score;
            }
            if (entry.flag == LowerBound) {
                alpha = std::max(alpha, entry.score);
            } else if (entry.flag == UpperBound) {
                beta = std::min(beta, entry.score);
            }
            if (alpha >= beta) {
                return entry.score;
            }
        }
    }

    const auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side)) {
            return board.side == rootSide ? -MateScore - depth : MateScore + depth;
        }
        return 0;
    }

    if (depth <= 0) {
        return quiescence(board, QuiescenceDepth, alpha, beta, rootSide);
    }

    if (board.side == rootSide) {
        int best = std::numeric_limits<int>::min();
        for (const Move& move : orderMoves(board, legal, rootSide)) {
            Board next = board;
            applyMove(next, move);
            best = std::max(best, search(next, depth - 1, alpha, beta, rootSide));
            alpha = std::max(alpha, best);
            if (alpha >= beta) {
                break;
            }
        }
        TranspositionEntry entry;
        entry.depth = depth;
        entry.score = best;
        entry.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
        {
            std::lock_guard<std::mutex> lock(transpositionMutex_);
            transposition_[key] = entry;
        }
        return best;
    }

    int best = std::numeric_limits<int>::max();
    for (const Move& move : orderMoves(board, legal, rootSide)) {
        Board next = board;
        applyMove(next, move);
        best = std::min(best, search(next, depth - 1, alpha, beta, rootSide));
        beta = std::min(beta, best);
        if (alpha >= beta) {
            break;
        }
    }
    TranspositionEntry entry;
    entry.depth = depth;
    entry.score = best;
    entry.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_);
        transposition_[key] = entry;
    }
    return best;
}

int LearningEngine::quiescence(Board board, int depth, int alpha, int beta, Color rootSide) const {
    if (shouldStop()) {
        return evaluator_.evaluate(board, rootSide);
    }

    const auto legal = generateLegalMoves(board, true);
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
    if (maximizing) {
        if (standPat >= beta) {
            return standPat;
        }
        alpha = std::max(alpha, standPat);
        int best = standPat;
        for (const Move& move : orderMoves(board, legal, rootSide)) {
            if (!inCheck && !isTacticalMove(board, move)) {
                continue;
            }
            Board next = board;
            applyMove(next, move);
            best = std::max(best, quiescence(next, depth - 1, alpha, beta, rootSide));
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
    beta = std::min(beta, standPat);
    int best = standPat;
    for (const Move& move : orderMoves(board, legal, rootSide)) {
        if (!inCheck && !isTacticalMove(board, move)) {
            continue;
        }
        Board next = board;
        applyMove(next, move);
        best = std::min(best, quiescence(next, depth - 1, alpha, beta, rootSide));
        beta = std::min(beta, best);
        if (alpha >= beta) {
            break;
        }
    }
    return best;
}

bool LearningEngine::canForceMate(Board board, int depth, Color attacker) const {
    if (shouldStop()) {
        return false;
    }
    const auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        return isKingAttacked(board, board.side) && board.side != attacker;
    }
    if (depth <= 0) {
        return false;
    }

    if (board.side == attacker) {
        for (const Move& move : orderMoves(board, legal, attacker)) {
            Board next = board;
            applyMove(next, move);
            if (canForceMate(next, depth - 1, attacker)) {
                return true;
            }
        }
        return false;
    }

    for (const Move& move : orderMoves(board, legal, attacker)) {
        Board next = board;
        applyMove(next, move);
        if (!canForceMate(next, depth - 1, attacker)) {
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
    Board next = board;
    applyMove(next, move);
    return isKingAttacked(next, next.side);
}

std::vector<int> LearningEngine::scoreRootMovesParallel(const Board& board, const std::vector<Move>& orderedMoves, Color rootSide) const {
    std::vector<int> scores(orderedMoves.size(), std::numeric_limits<int>::min());
    if (orderedMoves.empty()) {
        return scores;
    }

    const int workerCount = std::min<int>(std::max(1, threads_), static_cast<int>(orderedMoves.size()));
    if (workerCount <= 1) {
        for (std::size_t i = 0; i < orderedMoves.size(); ++i) {
            if (shouldStop()) {
                break;
            }
            Board next = board;
            applyMove(next, orderedMoves[i]);
            scores[i] = search(next, searchDepth_ - 1, -MateScore, MateScore, rootSide);
        }
        return scores;
    }

    std::atomic_size_t nextIndex{0};
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (int worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&, rootSide]() {
            while (!shouldStop()) {
                const std::size_t index = nextIndex.fetch_add(1);
                if (index >= orderedMoves.size()) {
                    break;
                }
                Board next = board;
                applyMove(next, orderedMoves[index]);
                scores[index] = search(next, searchDepth_ - 1, -MateScore, MateScore, rootSide);
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

std::vector<Move> LearningEngine::orderMoves(const Board& board, const std::vector<Move>& moves, Color rootSide) const {
    std::vector<Move> ordered = moves;
    std::stable_sort(ordered.begin(), ordered.end(), [&](const Move& left, const Move& right) {
        return moveOrderScore(board, left, rootSide) > moveOrderScore(board, right, rootSide);
    });
    return ordered;
}

int LearningEngine::moveOrderScore(const Board& board, const Move& move, Color rootSide) const {
    int score = 0;
    if (!move.isDrop() && board.squares[move.to] != 0) {
        const PieceType captured = typeOf(board.squares[move.to]);
        const PieceType moving = typeOf(board.squares[move.from]);
        score += 10000 + pieceValue(captured) * 10 - pieceValue(moving);
    }
    if (move.promote) {
        score += 6000;
    }
    if (move.isDrop()) {
        score += pieceValue(move.drop) / 3;
    }
    Board next = board;
    applyMove(next, move);
    if (isKingAttacked(next, next.side)) {
        score += 8000;
    }
    score += evaluator_.evaluate(next, rootSide) / 64;
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
    std::uint64_t hash = 1469598103934665603ull;
    auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(rootSide == Black ? 1u : 2u);
    mix(board.side == Black ? 3u : 4u);
    for (int piece : board.squares) {
        mix(static_cast<std::uint64_t>(piece + 32));
    }
    for (int i = 0; i < 15; ++i) {
        mix(static_cast<std::uint64_t>(board.blackHand[i] + 7 * i));
        mix(static_cast<std::uint64_t>(board.whiteHand[i] + 11 * i));
    }
    return hash;
}

} // namespace shogi
