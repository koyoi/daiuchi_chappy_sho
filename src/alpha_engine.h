#pragma once

#include "alpha_mcts.h"
#include "alpha_onnx_inference.h"
#include "mate_solver.h"
#include "opening_book.h"
#include "search_types.h"
#include "shogi_types.h"

#include <mutex>
#include <random>
#include <string>

namespace shogi {

class AlphaEngineWrapper {
public:
    AlphaEngineWrapper();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits,
                    const InfoCallback& infoCallback);
    MateResult searchMate(const Board& board, int timeLimitMs = 500);
    void clearGame();

    void setMaxMoveTimeMs(int ms);
    void setSimulations(int n);
    void setNNModel(const std::string& model);
    void setNNDevice(const std::string& device);
    void setBatchSize(int n);
    void setFPUReduction(double v);
    void setTemperatureDropMove(int m);
    void setBookEnabled(bool enabled) { bookEnabled_ = enabled; }
    void setReuseTree(bool enabled) { mcts_.setReuseTree(enabled); }
    bool loadBook(const std::string& path = "book.txt");
    bool ensureNN();
    bool nnReady() const { return nn_.isLoaded(); }
    const std::string& nnLastError() const { return nn_.lastError(); }
    const std::string& nnModelPath() const { return nn_.modelPath(); }
    SearchInfo lastSearchInfo() const;

    // For self-play training data
    const AlphaMCTSResult& lastMCTSResult() const { return lastMCTSResult_; }

private:
    AlphaOnnxInference nn_;
    AlphaMCTSEngine mcts_;
    MateSolver mateSolver_;
    OpeningBook book_;
    bool bookEnabled_ = true;
    int maxMoveTimeMs_ = 3000;
    int simulations_ = 1600;
    std::string nnModel_ = "alpha_model.onnx";
    std::string nnDevice_ = "auto";
    std::mt19937 rng_{std::random_device{}()};
    mutable SearchInfo lastSearchInfo_;
    AlphaMCTSResult lastMCTSResult_;
    mutable std::mutex infoMutex_;
};

} // namespace shogi
