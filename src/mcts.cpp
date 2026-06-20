#include "mcts.h"

#include "movegen.h"
#include "nn_bridge.h"
#include "position.h"

#include <algorithm>
#include <chrono>
#include <numeric>

namespace shogi {

MCTSEngine::MCTSEngine(NNBridge& nn) : nn_(nn) {}

MCTSNode* MCTSEngine::select(MCTSNode* node) const {
    while (node->expanded && !node->children.empty()) {
        int parentN = node->visitCount.load(std::memory_order_relaxed);
        MCTSNode* best = nullptr;
        double bestScore = -1e9;
        for (auto& child : node->children) {
            double score = child->ucb(config_.cPuct, parentN);
            if (score > bestScore) {
                bestScore = score;
                best = child.get();
            }
        }
        if (!best) break;
        node = best;
    }
    return node;
}

void MCTSEngine::expand(MCTSNode* node, const Board& board) {
    auto legal = generateLegalMoves(board);
    if (legal.empty()) {
        node->expanded = true;
        return;
    }

    NNOutput out = nn_.evaluate(board);

    double priorSum = 0.0;
    std::vector<double> priors(legal.size());
    for (int i = 0; i < legal.size(); ++i) {
        int idx = nn_.moveToIndex(legal[i], board.side);
        if (idx >= 0 && idx < static_cast<int>(out.policy.size())) {
            priors[i] = out.policy[idx];
        } else {
            priors[i] = 1e-6;
        }
        priorSum += priors[i];
    }
    if (priorSum > 0) {
        for (auto& p : priors) p /= priorSum;
    } else {
        double uniform = 1.0 / legal.size();
        for (auto& p : priors) p = uniform;
    }

    node->children.reserve(legal.size());
    for (int i = 0; i < legal.size(); ++i) {
        auto child = std::make_unique<MCTSNode>();
        child->move = legal[i];
        child->parent = node;
        child->prior = priors[i];
        node->children.push_back(std::move(child));
    }
    node->expanded = true;
}

void MCTSEngine::backpropagate(MCTSNode* node, double value) {
    double v = value;
    while (node) {
        node->visitCount.fetch_add(1, std::memory_order_relaxed);
        double old = node->valueSum.load(std::memory_order_relaxed);
        while (!node->valueSum.compare_exchange_weak(old, old + v, std::memory_order_relaxed)) {}
        v = -v;
        node = node->parent;
    }
}

std::vector<Move> MCTSEngine::extractPV(MCTSNode* root) const {
    std::vector<Move> pv;
    MCTSNode* node = root;
    while (node->expanded && !node->children.empty()) {
        MCTSNode* best = nullptr;
        int bestVisits = -1;
        for (auto& child : node->children) {
            int n = child->visitCount.load(std::memory_order_relaxed);
            if (n > bestVisits) {
                bestVisits = n;
                best = child.get();
            }
        }
        if (!best || bestVisits <= 0) break;
        pv.push_back(best->move);
        node = best;
    }
    return pv;
}

void MCTSEngine::addDirichletNoise(MCTSNode* node) {
    if (node->children.empty()) return;
    std::gamma_distribution<double> gamma(config_.dirichletAlpha, 1.0);
    std::vector<double> noise(node->children.size());
    double noiseSum = 0.0;
    for (auto& n : noise) {
        n = gamma(rng_);
        noiseSum += n;
    }
    if (noiseSum > 0) {
        for (auto& n : noise) n /= noiseSum;
    }
    double eps = config_.dirichletEpsilon;
    for (std::size_t i = 0; i < node->children.size(); ++i) {
        node->children[i]->prior =
            (1.0 - eps) * node->children[i]->prior + eps * noise[i];
    }
}

MCTSResult MCTSEngine::search(const Board& board, int timeLimitMs,
                               const InfoCallback& callback) {
    return search(board, config_, timeLimitMs, callback);
}

MCTSResult MCTSEngine::search(const Board& board, const MCTSConfig& config,
                               int timeLimitMs, const InfoCallback& callback) {
    config_ = config;
    auto startTime = std::chrono::steady_clock::now();
    auto deadline = startTime + std::chrono::milliseconds(timeLimitMs > 0 ? timeLimitMs : 100000);

    auto root = std::make_unique<MCTSNode>();
    expand(root.get(), board);

    if (root->children.empty()) {
        return MCTSResult{};
    }

    if (config_.addNoise) {
        addDirichletNoise(root.get());
    }

    int simCount = 0;
    int lastReportSim = 0;

    while (simCount < config_.simulations) {
        if (std::chrono::steady_clock::now() >= deadline) break;

        MCTSNode* leaf = select(root.get());

        Board leafBoard = board;
        std::vector<MCTSNode*> path;
        {
            MCTSNode* cur = leaf;
            while (cur != root.get()) {
                path.push_back(cur);
                cur = cur->parent;
            }
            std::reverse(path.begin(), path.end());
        }
        for (MCTSNode* n : path) {
            applyMove(leafBoard, n->move);
        }

        double value;
        if (!leaf->expanded) {
            expand(leaf, leafBoard);
            NNOutput out = nn_.evaluate(leafBoard);
            value = -out.value;
        } else {
            value = leaf->children.empty() ? -1.0 : 0.0;
        }

        backpropagate(leaf, value);
        ++simCount;

        if (callback && simCount - lastReportSim >= 100) {
            lastReportSim = simCount;
            auto now = std::chrono::steady_clock::now();
            int elapsed = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count());

            MCTSNode* bestChild = nullptr;
            int bestN = -1;
            for (auto& c : root->children) {
                int n = c->visitCount.load(std::memory_order_relaxed);
                if (n > bestN) { bestN = n; bestChild = c.get(); }
            }

            if (bestChild) {
                SearchInfo info;
                info.depth = simCount;
                info.scoreCp = static_cast<int>(bestChild->q() * 1000);
                info.nodes = simCount;
                info.timeMs = elapsed;
                info.bestMove = bestChild->move;
                info.hasBestMove = true;
                info.pv = extractPV(root.get());
                callback(info);
            }
        }
    }

    // Select best move: proportional to visit count with temperature
    // to ensure variety, especially with low sim counts
    MCTSNode* bestChild = nullptr;
    if (simCount < 30) {
        // Very few sims: pick proportionally to visit counts
        std::vector<double> weights;
        weights.reserve(root->children.size());
        for (auto& c : root->children) {
            weights.push_back(static_cast<double>(
                c->visitCount.load(std::memory_order_relaxed)));
        }
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int chosen = dist(rng_);
        bestChild = root->children[chosen].get();
    } else {
        int bestVisits = -1;
        for (auto& c : root->children) {
            int n = c->visitCount.load(std::memory_order_relaxed);
            if (n > bestVisits) { bestVisits = n; bestChild = c.get(); }
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    int elapsedMs = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());

    MCTSResult result;
    if (bestChild) {
        result.bestMove = bestChild->move;
        result.winRate = bestChild->q();
        result.pv = extractPV(root.get());
    }
    result.simulations = simCount;
    result.timeMs = elapsedMs;
    return result;
}

} // namespace shogi
