#pragma once

#include "mate_solver.h"
#include "nnue.h"
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

class NNUEEngine {
public:
    NNUEEngine();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback);

    void setSearchDepth(int depth);
    void setMaxMoveTimeMs(int milliseconds);
    void setThreads(int threads);
    int threadCount() const;
    void setBookEnabled(bool enabled);
    void setWarnOnNoWeights(bool enabled) { warnOnNoWeights_ = enabled; }
    void setReuseCache(bool enabled) { reuseCache_ = enabled; }
    void setHashSizeMB(int mb);
    bool loadBook(const std::string& path = "book.txt");
    bool loadNNUE(const std::string& path);
    const std::string& nnuePath() const { return nnuePath_; }
    SearchInfo lastSearchInfo() const;
    MateResult searchMate(const Board& board, int timeLimitMs = 500);

    void stop() { stopped_.store(true); }
    void prepareSearch() { stopped_.store(false); }

    void setParam(const std::string& name, int value);
    int getParam(const std::string& name) const;
    void setRootPruneWidth(int w) { rootPruneWidth_ = w; }
    void setMultiPV(int n) { multiPV_ = std::max(1, n); }
    int multiPV() const { return multiPV_; }
    const std::vector<RootMoveScore>& lastRootScores() const { return lastRootScores_; }

private:
    int search(Board& board, int depth, int ply, int alpha, int beta, bool allowNullMove = true, const Move& prevMove = Move{}, const Move& excludedMove = Move{}) const;
    int quiescence(Board& board, int depth, int ply, int alpha, int beta) const;
    std::vector<Move> extractPV(Board board, int maxDepth) const;
    void orderMoves(const Board& board, MoveList& moves, int ply, const Move& prevMove = Move{}) const;
    int moveOrderScore(const Board& board, const Move& move, int ply, const Move& ttMove, const Move& prevMove = Move{}) const;
    bool shouldStop() const;
    std::uint64_t boardHash(const Board& board) const;
    void setLastSearchInfo(const SearchInfo& info) const;
    int depthLimit() const;
    void clearSearchTables() const;
    void storeKiller(int ply, const Move& move) const;
    void updateHistory(Color side, const Move& move, int depth, bool good) const;
    void storeCounterMove(Color side, const Move& prevMove, const Move& counterMove) const;
    void workerSearch(const Board& board, const MoveList& legal, int threadId) const;

    int eval(const Board& board, int ply) const;

    struct TranspositionEntry {
        std::uint64_t key = 0;
        int depth = -1;
        int score = 0;
        int staticEval = 0;
        std::uint8_t flag = 0;
        std::uint8_t generation = 0;
        Move bestMove{};
    };

    static constexpr int BucketSize = 4;
    static constexpr int LockCount = 64;
    int ttBits_ = 20;
    int ttSize_ = 1 << 20;
    int ttMask_ = (1 << 20) - 1;
    static constexpr int MaxPly = 128;
    static constexpr int KillerSlots = 2;
    static constexpr int MaxLMRDepth = 64;
    static constexpr int MaxLMRMoves = 64;
    int lmrTable_[MaxLMRDepth][MaxLMRMoves]{};
    int seMinDepth_ = 8;
    int nmpMinDepth_ = 3;
    int nmpReduction_ = 3;
    int futilityMargin1_ = 600;
    int futilityMargin2_ = 1200;
    int aspirationWindow_ = 200;
    int iidMinDepth_ = 5;
    int deltaMargin_ = 2000;
    int qDepth_ = 6;
    int qCheckDepthMin_ = 4;

    NNUENetwork nnue_;
    MateSolver mateSolver_;
    OpeningBook book_;
    bool bookEnabled_ = true;
    bool warnOnNoWeights_ = true;
    std::string nnuePath_ = "nnue.bin";
    mutable std::vector<TranspositionEntry> transposition_;
    mutable std::array<std::mutex, LockCount> transpositionMutex_;
    mutable std::uint8_t ttGeneration_{0};
    bool reuseCache_ = true;
    bool isRecentGeneration(std::uint8_t gen) const {
        return gen == ttGeneration_ || (reuseCache_ && gen == static_cast<std::uint8_t>(ttGeneration_ - 1));
    }
    mutable std::chrono::steady_clock::time_point deadline_{};
    mutable std::atomic_bool stopped_{false};
    mutable std::atomic_uint64_t nodes_{0};
    mutable SearchInfo lastSearchInfo_{};
    mutable std::mutex lastSearchInfoMutex_;
    mutable std::array<std::array<Move, KillerSlots>, MaxPly> killers_{};
    mutable std::array<std::array<std::array<std::int16_t, BoardSize>, BoardSize>, 2> history_{};
    mutable std::array<std::array<std::array<Move, BoardSize>, BoardSize>, 2> counterMoves_{};
    static constexpr int ContHistPieces = 14;
    static constexpr int ContHistDim = ContHistPieces * BoardSize;
    mutable std::vector<std::int16_t> contHistory_;
    mutable std::int16_t captHistory_[ContHistPieces][BoardSize][ContHistPieces]{};
    mutable std::int16_t dropHistory_[2][7][BoardSize]{};
    int maxMoveTimeMs_ = 1000;
    int threads_ = 1;
    std::mt19937 rng_;
    int searchDepth_ = 0;
    int rootPruneWidth_ = 15;
    int multiPV_ = 1;
    mutable std::vector<RootMoveScore> lastRootScores_;
};

} // namespace shogi
