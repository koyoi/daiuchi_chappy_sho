#include "gpu_eval_engine.h"

#include "movegen.h"
#include "position.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace shogi {

namespace {

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

constexpr int MateScore = 100000;

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

GpuEvalEngine::GpuEvalEngine()
    : rng_(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())) {
    zobrist::init();
    gpu_.setEnabled(true);
}

Move GpuEvalEngine::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move GpuEvalEngine::chooseMove(const Board& board, const SearchLimits& limits) {
    return chooseMove(board, limits, InfoCallback{});
}

Move GpuEvalEngine::chooseMove(const Board& board, const SearchLimits& /*limits*/, const InfoCallback& infoCallback) {
    const auto searchStart = std::chrono::steady_clock::now();

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        SearchInfo info;
        info.depth = 0;
        info.nodes = 0;
        info.timeMs = 0;
        std::lock_guard<std::mutex> lock(infoMutex_);
        lastSearchInfo_ = info;
        return Move{};
    }

    std::vector<FeatureVector> features;
    features.reserve(legal.size());
    for (const Move& move : legal) {
        Board next = board;
        applyMove(next, move);
        features.push_back(evaluator_.extractFeatures(next, board.side));
    }

    std::vector<int> scores;
    if (!gpu_.score(features, scores)) {
        SearchInfo info;
        info.depth = 0;
        info.nodes = 0;
        info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
        std::lock_guard<std::mutex> lock(infoMutex_);
        lastSearchInfo_ = info;
        return legal.front();
    }

    int bestScore = std::numeric_limits<int>::min();
    std::vector<Move> bestMoves;
    for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
        const int adjustedScore = scores[i] - (openingSafety_ ? openingTrapPenalty(board, legal[i], board.side) : 0);
        if (adjustedScore > bestScore) {
            bestScore = adjustedScore;
            bestMoves.clear();
            bestMoves.push_back(legal[i]);
        } else if (adjustedScore == bestScore) {
            bestMoves.push_back(legal[i]);
        }
    }

    Move candidate;
    if (bestMoves.empty()) {
        candidate = legal.front();
    } else {
        std::uniform_int_distribution<std::size_t> dist(0, bestMoves.size() - 1);
        candidate = bestMoves[dist(rng_)];
    }

    Board tmp = board;
    applyMove(tmp, candidate);
    if (isKingAttacked(tmp, tmp.side)) {
        candidate = legal.front();
    }

    SearchInfo info;
    info.depth = 1;
    info.scoreCp = bestScore;
    info.nodes = static_cast<std::uint64_t>(legal.size());
    info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
    info.bestMove = candidate;
    info.hasBestMove = candidate.to >= 0;
    if (infoCallback) {
        infoCallback(info);
    }
    {
        std::lock_guard<std::mutex> lock(infoMutex_);
        lastSearchInfo_ = info;
    }
    return candidate;
}

void GpuEvalEngine::setMaxMoveTimeMs(int milliseconds) {
    maxMoveTimeMs_ = std::clamp(milliseconds, 50, 600000);
}

void GpuEvalEngine::setOpeningSafety(bool enabled) {
    openingSafety_ = enabled;
}

void GpuEvalEngine::setGpuPython(const std::string& python) {
    gpu_.setPython(python);
}

void GpuEvalEngine::setGpuScript(const std::string& script) {
    gpu_.setScript(script);
}

void GpuEvalEngine::setGpuModel(const std::string& model) {
    gpu_.setModel(model);
}

void GpuEvalEngine::setGpuDevice(const std::string& device) {
    gpu_.setDevice(device);
}

SearchInfo GpuEvalEngine::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(infoMutex_);
    return lastSearchInfo_;
}

} // namespace shogi
