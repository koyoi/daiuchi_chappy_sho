#include "alpha_mcts.h"

#include "movegen.h"
#include "position.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace shogi {

AlphaMCTSEngine::AlphaMCTSEngine(AlphaOnnxInference& nn) : nn_(nn) {}

double AlphaMCTSEngine::dynamicCPuct(int parentVisits) const {
    return std::log((1.0 + parentVisits + config_.cPuctBase) / config_.cPuctBase)
           + config_.cPuctInit;
}

AlphaMCTSNode* AlphaMCTSEngine::select(AlphaMCTSNode* node) const {
    while (node->expanded && !node->children.empty()) {
        int parentN = node->visitCount.load(std::memory_order_relaxed);
        double cPuct = dynamicCPuct(parentN);
        double parentQ = node->q();
        double fpuValue = parentQ - config_.fpuReduction;

        AlphaMCTSNode* best = nullptr;
        double bestScore = -1e9;
        for (auto& child : node->children) {
            double score = child->ucb(cPuct, parentN, fpuValue);
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

bool AlphaMCTSEngine::expandMoves(AlphaMCTSNode* node, const Board& board) {
    auto legal = generateLegalMoves(board);
    node->expanded = true;
    if (legal.empty()) {
        node->terminal = true;
        return false;
    }
    node->children.reserve(legal.size());
    double uniformPrior = 1.0 / static_cast<double>(legal.size());
    for (const auto& m : legal) {
        auto child = std::make_unique<AlphaMCTSNode>();
        child->move = m;
        child->parent = node;
        child->prior = uniformPrior;
        node->children.push_back(std::move(child));
    }
    return true;
}

void AlphaMCTSEngine::applyNNOutput(AlphaMCTSNode* node, const AlphaNNOutput& output,
                                     const Board& board) {
    if (node->children.empty()) return;
    double priorSum = 0.0;
    std::vector<double> priors(node->children.size());
    for (std::size_t i = 0; i < node->children.size(); ++i) {
        int idx = AlphaOnnxInference::moveToIndex(node->children[i]->move, board.side);
        if (idx >= 0 && idx < static_cast<int>(output.policy.size())) {
            priors[i] = output.policy[idx];
        } else {
            priors[i] = 1e-6;
        }
        priorSum += priors[i];
    }
    if (priorSum > 0) {
        for (auto& p : priors) p /= priorSum;
    } else {
        double uniform = 1.0 / static_cast<double>(node->children.size());
        for (auto& p : priors) p = uniform;
    }
    for (std::size_t i = 0; i < node->children.size(); ++i) {
        node->children[i]->prior = priors[i];
    }
}

void AlphaMCTSEngine::applyVirtualLoss(AlphaMCTSNode* node, int count) {
    AlphaMCTSNode* cur = node;
    while (cur) {
        cur->visitCount.fetch_add(count, std::memory_order_relaxed);
        double old = cur->valueSum.load(std::memory_order_relaxed);
        while (!cur->valueSum.compare_exchange_weak(old, old - count,
                                                     std::memory_order_relaxed)) {}
        cur = cur->parent;
    }
}

void AlphaMCTSEngine::removeVirtualLoss(AlphaMCTSNode* node, int count) {
    AlphaMCTSNode* cur = node;
    while (cur) {
        cur->visitCount.fetch_sub(count, std::memory_order_relaxed);
        double old = cur->valueSum.load(std::memory_order_relaxed);
        while (!cur->valueSum.compare_exchange_weak(old, old + count,
                                                     std::memory_order_relaxed)) {}
        cur = cur->parent;
    }
}

void AlphaMCTSEngine::backpropagate(AlphaMCTSNode* node, double value) {
    double v = value;
    while (node) {
        node->visitCount.fetch_add(1, std::memory_order_relaxed);
        double old = node->valueSum.load(std::memory_order_relaxed);
        while (!node->valueSum.compare_exchange_weak(old, old + v,
                                                      std::memory_order_relaxed)) {}
        v = -v;
        node = node->parent;
    }
}

std::vector<Move> AlphaMCTSEngine::extractPV(AlphaMCTSNode* root) const {
    std::vector<Move> pv;
    AlphaMCTSNode* node = root;
    while (node->expanded && !node->children.empty()) {
        AlphaMCTSNode* best = nullptr;
        int bestVisits = -1;
        for (auto& child : node->children) {
            int n = child->visitCount.load(std::memory_order_relaxed);
            if (n > bestVisits) { bestVisits = n; best = child.get(); }
        }
        if (!best || bestVisits <= 0) break;
        pv.push_back(best->move);
        node = best;
    }
    return pv;
}

void AlphaMCTSEngine::addDirichletNoise(AlphaMCTSNode* node) {
    if (node->children.empty()) return;
    std::gamma_distribution<double> gamma(config_.dirichletAlpha, 1.0);
    std::vector<double> noise(node->children.size());
    double noiseSum = 0.0;
    for (auto& n : noise) { n = gamma(rng_); noiseSum += n; }
    if (noiseSum > 0) {
        for (auto& n : noise) n /= noiseSum;
    }
    double eps = config_.dirichletEpsilon;
    for (std::size_t i = 0; i < node->children.size(); ++i) {
        node->children[i]->prior =
            (1.0 - eps) * node->children[i]->prior + eps * noise[i];
    }
}

AlphaMCTSResult AlphaMCTSEngine::search(const Board& board, int timeLimitMs,
                                         const InfoCallback& callback) {
    return search(board, config_, timeLimitMs, callback);
}

AlphaMCTSResult AlphaMCTSEngine::search(const Board& board, const AlphaMCTSConfig& config,
                                         int timeLimitMs, const InfoCallback& callback) {
    config_ = config;
    auto startTime = std::chrono::steady_clock::now();
    auto deadline = startTime + std::chrono::milliseconds(timeLimitMs > 0 ? timeLimitMs : 100000);

    std::unique_ptr<AlphaMCTSNode> root;
    int reuseVisits = 0;

    if (reuseTree_ && retainedTree_ && retainedTree_->expanded) {
        for (auto& child : retainedTree_->children) {
            if (!child->expanded) continue;
            for (auto& grandchild : child->children) {
                Board tmp = retainedBoard_;
                applyMove(tmp, child->move);
                applyMove(tmp, grandchild->move);
                if (tmp.hash == board.hash) {
                    reuseVisits = grandchild->visitCount.load(std::memory_order_relaxed);
                    root = std::move(grandchild);
                    root->parent = nullptr;
                    goto reuse_done;
                }
            }
        }
    }
reuse_done:
    retainedTree_.reset();

    if (!root) {
        root = std::make_unique<AlphaMCTSNode>();
        if (!expandMoves(root.get(), board)) {
            return AlphaMCTSResult{};
        }
        {
            AlphaNNOutput rootOut = nn_.evaluate(board);
            applyNNOutput(root.get(), rootOut, board);
        }
    } else {
        std::cout << "info string reusing tree (" << reuseVisits << " sims)" << std::endl;
        if (!root->expanded) {
            if (!expandMoves(root.get(), board)) {
                return AlphaMCTSResult{};
            }
            AlphaNNOutput rootOut = nn_.evaluate(board);
            applyNNOutput(root.get(), rootOut, board);
        }
    }

    if (config_.addNoise) {
        addDirichletNoise(root.get());
    }

    int simCount = 0;
    int lastReportSim = 0;

    struct PendingEval {
        AlphaMCTSNode* leaf;
        Board board;
        bool terminal;
        int batchIndex;
    };

    while (simCount < config_.simulations) {
        if (std::chrono::steady_clock::now() >= deadline) break;

        // Early termination: best move has >90% of visits
        if (simCount > config_.simulations / 4) {
            int totalVisits = root->visitCount.load(std::memory_order_relaxed);
            int bestN = 0;
            for (auto& c : root->children) {
                int n = c->visitCount.load(std::memory_order_relaxed);
                if (n > bestN) bestN = n;
            }
            if (totalVisits > 0 && bestN * 10 > totalVisits * 9) break;
        }

        std::vector<PendingEval> pending;
        std::vector<Board> evalBoards;

        int batchTarget = std::min(config_.batchSize, config_.simulations - simCount);

        for (int b = 0; b < batchTarget; ++b) {
            AlphaMCTSNode* leaf = select(root.get());

            Board leafBoard = board;
            std::vector<AlphaMCTSNode*> path;
            {
                AlphaMCTSNode* cur = leaf;
                while (cur != root.get()) {
                    path.push_back(cur);
                    cur = cur->parent;
                }
                std::reverse(path.begin(), path.end());
            }
            for (AlphaMCTSNode* n : path) {
                applyMove(leafBoard, n->move);
            }

            if (leaf->expanded) {
                double val = leaf->terminal ? 1.0 : -leaf->q();
                pending.push_back({leaf, leafBoard, true, -1});
                backpropagate(leaf, val);
                ++simCount;
                continue;
            }

            bool hasChildren = expandMoves(leaf, leafBoard);
            if (!hasChildren) {
                pending.push_back({leaf, leafBoard, true, -1});
                backpropagate(leaf, 1.0);
                ++simCount;
                continue;
            }

            applyVirtualLoss(leaf, config_.virtualLoss);
            int idx = static_cast<int>(evalBoards.size());
            evalBoards.push_back(leafBoard);
            pending.push_back({leaf, leafBoard, false, idx});
        }

        if (!evalBoards.empty()) {
            std::vector<AlphaNNOutput> outputs;
            if (evalBoards.size() == 1) {
                outputs.push_back(nn_.evaluate(evalBoards[0]));
            } else {
                outputs = nn_.evaluateBatch(evalBoards);
            }

            for (auto& p : pending) {
                if (p.terminal) continue;
                removeVirtualLoss(p.leaf, config_.virtualLoss);
                applyNNOutput(p.leaf, outputs[p.batchIndex], p.board);
                backpropagate(p.leaf, -outputs[p.batchIndex].value);
                ++simCount;
            }
        }

        if (callback && simCount - lastReportSim >= 100) {
            lastReportSim = simCount;
            auto now = std::chrono::steady_clock::now();
            int elapsed = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count());

            AlphaMCTSNode* bestChild = nullptr;
            int bestN = -1;
            for (auto& c : root->children) {
                int n = c->visitCount.load(std::memory_order_relaxed);
                if (n > bestN) { bestN = n; bestChild = c.get(); }
            }

            if (bestChild) {
                SearchInfo info;
                info.pv = extractPV(root.get());
                info.depth = static_cast<int>(info.pv.size());
                info.scoreCp = static_cast<int>(bestChild->q() * 1000);
                info.nodes = simCount;
                info.timeMs = elapsed;
                info.bestMove = bestChild->move;
                info.hasBestMove = true;
                callback(info);
            }
        }
    }

    // Select best move
    AlphaMCTSNode* bestChild = nullptr;
    if (board.moveNumber <= config_.temperatureDropMove) {
        // Stochastic selection proportional to visit count
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

    AlphaMCTSResult result;
    if (bestChild) {
        result.bestMove = bestChild->move;
        result.winRate = bestChild->q();
    }

    // Compute root WDL from best child
    // Since we alternate sign, use root's perspective
    int totalRootVisits = root->visitCount.load(std::memory_order_relaxed);
    if (totalRootVisits > 0) {
        double rootQ = root->q();
        result.wdl[0] = std::max(0.0, std::min(1.0, (1.0 + rootQ) / 2.0));
        result.wdl[2] = std::max(0.0, std::min(1.0, (1.0 - rootQ) / 2.0));
        result.wdl[1] = 1.0 - result.wdl[0] - result.wdl[2];
    }

    // Collect visit distribution for training data
    for (auto& c : root->children) {
        int n = c->visitCount.load(std::memory_order_relaxed);
        if (n > 0) {
            result.visitDistribution.emplace_back(c->move, n);
        }
    }

    result.simulations = simCount;
    result.timeMs = elapsedMs;
    result.pv = extractPV(root.get());

    if (reuseTree_) {
        retainedBoard_ = board;
        retainedTree_ = std::move(root);
    }

    return result;
}

} // namespace shogi
