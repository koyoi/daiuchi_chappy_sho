#pragma once

#include "alpha_onnx_inference.h"
#include "search_types.h"
#include "shogi_types.h"

#include <atomic>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace shogi {

struct AlphaMCTSNode {
    Move move{};
    AlphaMCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<AlphaMCTSNode>> children;

    std::atomic<int> visitCount{0};
    std::atomic<double> valueSum{0.0};
    double prior = 0.0;
    bool expanded = false;
    bool terminal = false;

    double q() const {
        int n = visitCount.load(std::memory_order_relaxed);
        return n > 0 ? valueSum.load(std::memory_order_relaxed) / n : 0.0;
    }

    double ucb(double cPuct, int parentVisits, double fpuValue) const {
        int n = visitCount.load(std::memory_order_relaxed);
        double qVal = n > 0 ? q() : fpuValue;
        return qVal + cPuct * prior * std::sqrt(static_cast<double>(parentVisits)) / (1.0 + n);
    }
};

struct AlphaMCTSConfig {
    int simulations = 1600;
    double cPuctBase = 19652.0;
    double cPuctInit = 2.5;
    double fpuReduction = 0.2;
    double dirichletAlpha = 0.15;
    double dirichletEpsilon = 0.25;
    int batchSize = 16;
    int virtualLoss = 3;
    bool addNoise = true;
    int temperatureDropMove = 30;
};

struct AlphaMCTSResult {
    Move bestMove{};
    double winRate = 0.0;
    std::array<double, 3> wdl = {0.0, 0.0, 0.0};
    int simulations = 0;
    int timeMs = 0;
    std::vector<Move> pv;
    std::vector<std::pair<Move, int>> visitDistribution;
};

class AlphaMCTSEngine {
public:
    explicit AlphaMCTSEngine(AlphaOnnxInference& nn);

    AlphaMCTSResult search(const Board& board, int timeLimitMs,
                           const InfoCallback& callback);
    AlphaMCTSResult search(const Board& board, const AlphaMCTSConfig& config,
                           int timeLimitMs, const InfoCallback& callback);

    void setSimulations(int n) { config_.simulations = n; }
    void setBatchSize(int n) { config_.batchSize = std::max(1, n); }
    void setFPUReduction(double v) { config_.fpuReduction = v; }
    void setTemperatureDropMove(int m) { config_.temperatureDropMove = m; }

private:
    double dynamicCPuct(int parentVisits) const;
    AlphaMCTSNode* select(AlphaMCTSNode* node) const;
    bool expandMoves(AlphaMCTSNode* node, const Board& board);
    void applyNNOutput(AlphaMCTSNode* node, const AlphaNNOutput& output, const Board& board);
    void applyVirtualLoss(AlphaMCTSNode* node, int count);
    void removeVirtualLoss(AlphaMCTSNode* node, int count);
    void backpropagate(AlphaMCTSNode* node, double value);
    std::vector<Move> extractPV(AlphaMCTSNode* root) const;
    void addDirichletNoise(AlphaMCTSNode* node);

    AlphaOnnxInference& nn_;
    AlphaMCTSConfig config_;
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace shogi
