#pragma once

#include "search_types.h"
#include "shogi_types.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace shogi {

class NNBridge;

struct MCTSNode {
    Move move{};
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;

    std::atomic<int> visitCount{0};
    std::atomic<double> valueSum{0.0};
    double prior = 0.0;
    bool expanded = false;

    double q() const {
        int n = visitCount.load(std::memory_order_relaxed);
        return n > 0 ? valueSum.load(std::memory_order_relaxed) / n : 0.0;
    }

    double ucb(double cPuct, int parentVisits) const {
        int n = visitCount.load(std::memory_order_relaxed);
        return q() + cPuct * prior * std::sqrt(static_cast<double>(parentVisits)) / (1.0 + n);
    }
};

struct MCTSConfig {
    int simulations = 800;
    double cPuct = 1.5;
    double dirichletAlpha = 0.15;
    double dirichletEpsilon = 0.25;
    int batchSize = 8;
    bool addNoise = true;
};

struct MCTSResult {
    Move bestMove{};
    double winRate = 0.0;
    int simulations = 0;
    int timeMs = 0;
    std::vector<Move> pv;
};

class MCTSEngine {
public:
    explicit MCTSEngine(NNBridge& nn);

    MCTSResult search(const Board& board, int timeLimitMs, const InfoCallback& callback);
    MCTSResult search(const Board& board, const MCTSConfig& config, int timeLimitMs,
                      const InfoCallback& callback);

    void setSimulations(int n) { config_.simulations = n; }

private:
    MCTSNode* select(MCTSNode* node) const;
    void expand(MCTSNode* node, const Board& board);
    void backpropagate(MCTSNode* node, double value);
    std::vector<Move> extractPV(MCTSNode* root) const;
    void addDirichletNoise(MCTSNode* node);

    NNBridge& nn_;
    MCTSConfig config_;
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace shogi
