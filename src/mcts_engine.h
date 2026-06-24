#pragma once

#include "mate_solver.h"
#include "mcts.h"
#include "nn_bridge.h"
#include "opening_book.h"
#include "search_types.h"
#include "shogi_types.h"

#include <mutex>
#include <random>
#include <string>

namespace shogi {

class MCTSEngineWrapper {
public:
    MCTSEngineWrapper();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback);
    MateResult searchMate(const Board& board, int timeLimitMs = 500);
    void recordMove(const Board& before, const Move& move, bool engineMove);
    void finishGame(int engineResult, Color engineSide);
    void clearGame();

    void setMaxMoveTimeMs(int milliseconds);
    void setSimulations(int n);
    void setNNPython(const std::string& python);
    void setNNScript(const std::string& script);
    void setNNModel(const std::string& model);
    void setNNDevice(const std::string& device);
    void setBatchSize(int n);
    void setBookEnabled(bool enabled) { bookEnabled_ = enabled; }
    void setWarnOnNoModel(bool enabled) { warnOnNoModel_ = enabled; }
    bool loadBook(const std::string& path = "book.txt");
    bool ensureNN();
    bool nnReady() const { return nn_.isReady(); }
    const std::string& nnLastError() const;
    const std::string& nnModelPath() const;
    SearchInfo lastSearchInfo() const;

private:
    NNBridge nn_;
    MCTSEngine mcts_;
    MateSolver mateSolver_;
    OpeningBook book_;
    bool bookEnabled_ = true;
    bool warnOnNoModel_ = true;
    int maxMoveTimeMs_ = 3000;
    int simulations_ = 800;
    std::mt19937 rng_{std::random_device{}()};
    mutable SearchInfo lastSearchInfo_;
    mutable std::mutex infoMutex_;
};

} // namespace shogi
