#pragma once

#include "evaluation.h"
#include "learning.h"
#include "mate_solver.h"
#include "opening_book.h"
#include "search_types.h"
#include "shogi_types.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace shogi {

class LearningEngine {
public:
    LearningEngine();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback);
    void recordMove(const Board& before, const Move& move, bool engineMove);
    void finishGame(int engineResult, Color engineSide);
    void clearGame();

    void setLearningEnabled(bool enabled);
    void setRecordOnly(bool recordOnly);
    void setSearchDepth(int depth);
    void setMaxMoveTimeMs(int milliseconds);
    void setHeavyEvaluation(bool enabled);
    void setOpeningSafety(bool enabled);
    void setThreads(int threads);
    int threadCount() const;
    void setWeightsPath(const std::string& path);
    void setTrainingDataPath(const std::string& path);
    bool loadWeights();
    bool loadMlpWeights(const std::string& path);
    void setUseMlp(bool use) { evaluator_.setUseMlp(use); }
    const std::string& weightsPath() const;
    const std::string& mlpWeightsPath() const { return mlpWeightsPath_; }
    void setRootPruneWidth(int width);
    void setBookEnabled(bool enabled);
    void setWarnOnNoWeights(bool enabled) { warnOnNoWeights_ = enabled; }
    bool loadBook(const std::string& path = "book.txt");
    SearchInfo lastSearchInfo() const;
    MateResult searchMate(const Board& board, int timeLimitMs = 500);

private:
    int search(Board& board, int depth, int ply, int alpha, int beta, Color rootSide, bool allowNullMove = true, const Move& prevMove = Move{}) const;
    std::vector<Move> extractPV(Board board, Color rootSide, int maxDepth) const;
    int quiescence(Board& board, int depth, int ply, int alpha, int beta, Color rootSide) const;
    bool canForceMate(Board& board, int depth, Color attacker) const;
    bool isTacticalMove(const Board& board, const Move& move) const;
    std::vector<int> scoreRootMovesParallel(
        const Board& board,
        const MoveList& orderedMoves,
        const std::vector<int>& openingPenalties,
        Color rootSide,
        int depth,
        int pruneWidth,
        int aspirationAlpha,
        int aspirationBeta,
        const std::chrono::steady_clock::time_point& searchStart,
        const InfoCallback& infoCallback) const;
    void orderMoves(const Board& board, MoveList& moves, Color rootSide, int ply, const Move& prevMove = Move{}) const;
    int moveOrderScore(const Board& board, const Move& move, Color rootSide, int ply, const Move& ttMove, const Move& prevMove = Move{}) const;
    bool shouldStop() const;
    std::uint64_t boardHash(const Board& board, Color rootSide) const;
    void setLastSearchInfo(const SearchInfo& info) const;
    int depthLimit() const;
    void clearSearchTables() const;
    void storeKiller(int ply, const Move& move) const;
    void updateHistory(Color side, const Move& move, int depth, bool good) const;
    void storeCounterMove(Color side, const Move& prevMove, const Move& counterMove) const;
    void analyzeSubGoals(const Board& board, Color rootSide) const;

    struct TranspositionEntry {
        std::uint64_t key = 0;
        int depth = -1;
        int score = 0;
        std::uint8_t flag = 0;
        std::uint8_t generation = 0;
        Move bestMove{};
    };

    static constexpr int TTBits = 20;
    static constexpr int TTSize = 1 << TTBits;
    static constexpr int TTMask = TTSize - 1;
    static constexpr int LockCount = 64;
    static constexpr int MaxPly = 128;
    static constexpr int KillerSlots = 2;
    static constexpr int LMRFullDepthMoves = 4;
    static constexpr int LMRMinDepth = 3;
    static constexpr int NMPMinDepth = 3;
    static constexpr int NMPReduction = 3;
    static constexpr int FutilityMargin1 = 400;
    static constexpr int FutilityMargin2 = 900;
    static constexpr int AspirationWindow = 50;
    static constexpr int IIDMinDepth = 5;
    static constexpr int MaxExtensions = 16;
    static constexpr int SubGoalDepth = 3;
    static constexpr int SubGoalThreshold = 200;
    static constexpr int SubGoalCaptureBonus = 2000;

    Evaluator evaluator_;
    OnlineLearner learner_;
    MateSolver mateSolver_;
    OpeningBook book_;
    bool bookEnabled_ = true;
    bool warnOnNoWeights_ = true;
    std::string mlpWeightsPath_ = "mlp.weights";
    mutable std::vector<TranspositionEntry> transposition_;
    mutable std::array<std::mutex, LockCount> transpositionMutex_;
    mutable std::uint8_t ttGeneration_{0};
    mutable std::chrono::steady_clock::time_point deadline_{};
    mutable std::atomic_bool stopped_{false};
    mutable std::atomic_uint64_t nodes_{0};
    mutable SearchInfo lastSearchInfo_{};
    mutable std::mutex lastSearchInfoMutex_;
    mutable std::array<std::array<Move, KillerSlots>, MaxPly> killers_{};
    mutable std::array<std::array<std::array<std::int16_t, BoardSize>, BoardSize>, 2> history_{};
    mutable std::array<std::array<std::array<Move, BoardSize>, BoardSize>, 2> counterMoves_{};
    mutable std::array<bool, PieceTypeCount> targetPieces_{};
    int maxMoveTimeMs_ = 1000;
    int threads_ = 1;
    std::mt19937 rng_;
    int searchDepth_ = 0;
    bool openingSafety_ = true;
    int rootPruneWidth_ = 15;
};

} // namespace shogi
