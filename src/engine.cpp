#include "engine.h"

#include "movegen.h"
#include "position.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

constexpr int MateScore = 29000;       // 詰みの評価値（十分大きな値）
constexpr int QuiescenceDepth = 2;     // 静止探索の最大深さ
constexpr int ExactScore = 0;          // 置換表: 正確な評価値
constexpr int LowerBound = 1;          // 置換表: 下界（βカット発生）
constexpr int UpperBound = 2;          // 置換表: 上界（αカット発生）

// 駒種ごとの静的な価値（歩=100基準）
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

int chebyshevDistance(int left, int right) {
    return std::max(std::abs(fileOf(left) - fileOf(right)), std::abs(rankOf(left) - rankOf(right)));
}

// 序盤の安全性ペナルティの強さを手数に応じて減衰させる (0〜100%)
// 24手目までは100%適用、56手目以降は無効
int openingSafetyScale(int moveNumber) {
    if (moveNumber >= 56) {
        return 0;
    }
    if (moveNumber <= 24) {
        return 100;
    }
    return std::max(0, (56 - moveNumber) * 100 / 32);
}

// 成ることで得られる駒価値の増分
int promotionGain(PieceType type) {
    if (!canPromote(type)) {
        return 0;
    }
    return std::max(0, pieceValue(promote(type)) - pieceValue(type));
}

// 自陣の駒が相手に取られやすい状態にあるかを評価するペナルティ
// 利きの数で攻め駒と守り駒のバランスを見る
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

// 玉の周囲8マス+玉自身への相手の利きを数え、玉の危険度を評価
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

// 相手の次の一手でどれだけ脅威があるかを評価
// 駒取り・成り・玉への接近・王手・詰みの可能性を総合的にスコア化
int opponentImmediateThreatScore(const Board& board, Color targetSide) {
    if (board.side == targetSide) {
        return 0;
    }
    const int king = findKing(board, targetSide);
    const auto replies = generateLegalMoves(board, true);
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

        Board next = board;
        applyMove(next, reply);
        if (isKingAttacked(next, targetSide)) {
            score += 1100;
            ++checks;
            if (generateLegalMoves(next, true).empty()) {
                return MateScore;
            }
        }

        if (score > 0) {
            best = std::max(best, score);
            accumulated += std::min(score, 1400) / 5;
        }
    }
    return best + accumulated + checks * 160;
}

// 序盤で危険な手（相手の脅威を招く手）にペナルティを与える
// 脅威スコア・駒の取られやすさ・玉の露出度を手数に応じた重みで合算
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

LearningEngine::LearningEngine()
    : learner_(evaluator_),
      transposition_(TTSize),
      rng_(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())) {
    zobrist::init();
    threads_ = defaultThreadCount();
    learner_.loadWeights();
}

Move LearningEngine::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move LearningEngine::chooseMove(const Board& board, const SearchLimits& limits) {
    return chooseMove(board, limits, InfoCallback{});
}

// 指し手選択のメイン関数
// 反復深化(iterative deepening)で探索深さを1から順に深くし、
// 制限時間内で最も良い手を選ぶ
Move LearningEngine::chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback) {
    // 探索状態の初期化
    const auto searchStart = std::chrono::steady_clock::now();
    stopped_.store(false);
    nodes_.store(0);
    ++ttGeneration_;
    // 持ち時間の設定（50ms〜600秒にクランプ）
    const int requestedMoveTime = limits.moveTimeMs > 0 ? limits.moveTimeMs : maxMoveTimeMs_;
    const int moveTime = std::clamp(requestedMoveTime, 50, 600000);
    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(moveTime);

    // 合法手がなければ投了（指し手なし）
    const auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        SearchInfo info;
        info.depth = 0;
        info.nodes = nodes_.load();
        info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
        setLastSearchInfo(info);
        return Move{};
    }

    // GPU評価が利用可能ならそちらで選択（ニューラルネット推論）
    Move gpuMove;
    if (chooseMoveByGpu(board, legal, gpuMove)) {
        SearchInfo info;
        info.depth = 1;
        info.nodes = static_cast<std::uint64_t>(legal.size());
        info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
        info.bestMove = gpuMove;
        info.hasBestMove = true;
        setLastSearchInfo(info);
        return gpuMove;
    }

    // 合法手を有望な順に並べ替え（探索効率のため）
    const Color rootSide = board.side;
    const std::vector<Move> orderedMoves = orderMoves(board, legal, rootSide);

    // 序盤の安全性ペナルティを各手に対して事前計算
    std::vector<int> openingPenalties(orderedMoves.size(), 0);
    if (openingSafety_) {
        for (std::size_t i = 0; i < orderedMoves.size(); ++i) {
            openingPenalties[i] = openingTrapPenalty(board, orderedMoves[i], rootSide);
        }
    }

    // 反復深化ループ: 深さ1から順に探索し、時間切れまで繰り返す
    int completedDepth = 0;
    int bestScore = std::numeric_limits<int>::min();
    std::vector<Move> bestMoves;
    const int maxDepth = depthLimit();
    for (int depth = 1; depth <= maxDepth && !shouldStop(); ++depth) {
        const std::uint64_t nodesBeforeDepth = nodes_.load();
        // この深さで全ルート手を並列評価
        const std::vector<int> scores = scoreRootMovesParallel(board, orderedMoves, openingPenalties, rootSide, depth, searchStart, infoCallback);

        // 時間切れでノードを1つも探索できなかったら、この深さの結果は破棄
        if (shouldStop() && nodes_.load() == nodesBeforeDepth) {
            break;
        }

        // この深さでの最善手を選出（同点なら複数保持）
        int depthBestScore = std::numeric_limits<int>::min();
        std::vector<Move> depthBestMoves;
        for (std::size_t i = 0; i < orderedMoves.size(); ++i) {
            const int score = scores[i];
            if (score == std::numeric_limits<int>::min()) {
                continue;
            }
            if (score > depthBestScore) {
                depthBestScore = score;
                depthBestMoves.clear();
                depthBestMoves.push_back(orderedMoves[i]);
            } else if (score == depthBestScore) {
                depthBestMoves.push_back(orderedMoves[i]);
            }
        }
        // 探索が完了した深さの結果で最善手を更新
        if (!depthBestMoves.empty()) {
            completedDepth = depth;
            bestScore = depthBestScore;
            bestMoves = depthBestMoves;
            SearchInfo info;
            info.depth = completedDepth;
            info.scoreCp = std::clamp(bestScore, -MateScore, MateScore);
            info.nodes = nodes_.load();
            info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
            info.bestMove = bestMoves.front();
            info.hasBestMove = true;
            setLastSearchInfo(info);
            if (infoCallback) {
                infoCallback(info);
            }
        }
    }

    // 最善手が同点で複数あればランダムに1つ選択
    if (bestMoves.empty()) {
        bestMoves.push_back(legal.front());
    }
    std::uniform_int_distribution<std::size_t> dist(0, bestMoves.size() - 1);
    const Move selected = bestMoves[dist(rng_)];
    SearchInfo info;
    info.depth = completedDepth;
    info.scoreCp = bestScore == std::numeric_limits<int>::min() ? 0 : std::clamp(bestScore, -MateScore, MateScore);
    info.nodes = nodes_.load();
    info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count());
    info.bestMove = selected;
    info.hasBestMove = true;
    setLastSearchInfo(info);
    return selected;
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
    searchDepth_ = std::clamp(depth, 0, 128);
}

void LearningEngine::setMaxMoveTimeMs(int milliseconds) {
    maxMoveTimeMs_ = std::clamp(milliseconds, 50, 600000);
}

void LearningEngine::setHeavyEvaluation(bool enabled) {
    evaluator_.setHeavyFeatures(enabled);
}

void LearningEngine::setOpeningSafety(bool enabled) {
    openingSafety_ = enabled;
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

SearchInfo LearningEngine::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    return lastSearchInfo_;
}

// GPU(ニューラルネット)による手の選択
// 全合法手の局面特徴量を抽出してGPUでスコアリングし、最善手を選ぶ
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
        const int adjustedScore = scores[i] - (openingSafety_ ? openingTrapPenalty(board, legal[i], board.side) : 0);
        if (adjustedScore > bestScore) {
            bestScore = adjustedScore;
            bestMoves.clear();
            bestMoves.push_back(legal[i]);
        } else if (adjustedScore == bestScore) {
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

// αβ探索 (minimax + αβ枝刈り)
// rootSide視点のスコアを返す。自分の手番ではmax、相手の手番ではmin。
// 置換表を使って同一局面の再探索を回避する。
int LearningEngine::search(Board board, int depth, int alpha, int beta, Color rootSide) const {
    nodes_.fetch_add(1);
    if (shouldStop()) {
        return evaluator_.evaluate(board, rootSide);
    }

    // 置換表の参照: 以前に同じ局面をより深く探索済みならその結果を再利用
    const int alphaOriginal = alpha;
    const int betaOriginal = beta;
    const std::uint64_t key = boardHash(board, rootSide);
    const int ttIndex = static_cast<int>(key & TTMask);
    const int lockIndex = static_cast<int>((key >> TTBits) % LockCount);
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        const TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.key == key && slot.generation == ttGeneration_ && slot.depth >= depth) {
            if (slot.flag == ExactScore) {
                return slot.score;
            }
            if (slot.flag == LowerBound) {
                alpha = std::max(alpha, slot.score);
            } else if (slot.flag == UpperBound) {
                beta = std::min(beta, slot.score);
            }
            if (alpha >= beta) {
                return slot.score;
            }
        }
    }

    // 終端ノード: 合法手なし → 詰み or ステイルメイト
    const auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        if (isKingAttacked(board, board.side)) {
            // 詰まされた側が自分なら大きなマイナス、相手なら大きなプラス
            return board.side == rootSide ? -MateScore - depth : MateScore + depth;
        }
        return 0;
    }

    // 深さ0に達したら静止探索へ移行（駒取りなど激しい変化を読み切る）
    if (depth <= 0) {
        return quiescence(board, QuiescenceDepth, alpha, beta, rootSide);
    }

    // 自分の手番: スコアを最大化 (max node)
    if (board.side == rootSide) {
        int best = std::numeric_limits<int>::min();
        for (const Move& move : orderMoves(board, legal, rootSide)) {
            Board next = board;
            applyMove(next, move);
            best = std::max(best, search(next, depth - 1, alpha, beta, rootSide));
            alpha = std::max(alpha, best);
            if (alpha >= beta) {
                break; // βカット: これ以上探索しても相手が許さない
            }
        }
        // 探索結果を置換表に保存
        {
            std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
            TranspositionEntry& slot = transposition_[ttIndex];
            if (slot.generation != ttGeneration_ || depth >= slot.depth) {
                slot.key = key;
                slot.depth = depth;
                slot.score = best;
                slot.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
                slot.generation = ttGeneration_;
            }
        }
        return best;
    }

    // 相手の手番: スコアを最小化 (min node)
    int best = std::numeric_limits<int>::max();
    for (const Move& move : orderMoves(board, legal, rootSide)) {
        Board next = board;
        applyMove(next, move);
        best = std::min(best, search(next, depth - 1, alpha, beta, rootSide));
        beta = std::min(beta, best);
        if (alpha >= beta) {
            break; // αカット: これ以上探索しても自分に有利にならない
        }
    }
    {
        std::lock_guard<std::mutex> lock(transpositionMutex_[lockIndex]);
        TranspositionEntry& slot = transposition_[ttIndex];
        if (slot.generation != ttGeneration_ || depth >= slot.depth) {
            slot.key = key;
            slot.depth = depth;
            slot.score = best;
            slot.flag = best <= alphaOriginal ? UpperBound : (best >= betaOriginal ? LowerBound : ExactScore);
            slot.generation = ttGeneration_;
        }
    }
    return best;
}

// 静止探索 (quiescence search)
// 通常探索の葉ノードで、駒取り・成り・王手などの激しい手だけを追加で読む。
// 「水平線効果」を緩和し、駒交換の途中で評価が切れることを防ぐ。
int LearningEngine::quiescence(Board board, int depth, int alpha, int beta, Color rootSide) const {
    nodes_.fetch_add(1);
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

    // stand pat: 何もしない場合の現局面の評価値
    int standPat = evaluator_.evaluate(board, rootSide);
    if (depth <= 0) {
        return standPat;
    }

    const bool maximizing = board.side == rootSide;
    const bool inCheck = isKingAttacked(board, board.side);

    // 自分の手番: stand patがβ以上なら枝刈り（十分良い局面）
    if (maximizing) {
        if (standPat >= beta) {
            return standPat;
        }
        alpha = std::max(alpha, standPat);
        int best = standPat;
        for (const Move& move : orderMoves(board, legal, rootSide)) {
            // 王手がかかっていなければ、戦術的な手(駒取り・成り・王手)のみ探索
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

    // 相手の手番: stand patがα以下なら枝刈り
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

// 詰み探索: attacker側が強制的に詰ませられるかを判定
// 攻め方はどれか1手で詰みに至ればtrue、受け方は全ての応手で詰まなければfalse
bool LearningEngine::canForceMate(Board board, int depth, Color attacker) const {
    nodes_.fetch_add(1);
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

// 戦術的な手かどうかを判定（駒取り・成り・王手のいずれか）
// 静止探索で読む手を絞り込むために使う
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

// ルートの全合法手を並列に評価する
// 各手に対してsearch()を呼び、序盤ペナルティを差し引いたスコアを返す
// マルチスレッド対応: ワーカースレッドが手を1つずつ取り出して探索
std::vector<int> LearningEngine::scoreRootMovesParallel(
    const Board& board,
    const std::vector<Move>& orderedMoves,
    const std::vector<int>& openingPenalties,
    Color rootSide,
    int depth,
    const std::chrono::steady_clock::time_point& searchStart,
    const InfoCallback& infoCallback) const {
    std::vector<int> scores(orderedMoves.size(), std::numeric_limits<int>::min());
    if (orderedMoves.empty()) {
        return scores;
    }

    std::mutex bestMutex;
    std::mutex emitMutex;
    int bestScore = std::numeric_limits<int>::min();
    Move bestMove;
    bool hasBestMove = false;
    auto lastEmit = searchStart - std::chrono::milliseconds(1000);

    // 探索途中の情報(現在の最善手・ノード数等)をコールバックで通知
    auto publish = [&](std::size_t index, int score, bool force) {
        SearchInfo info;
        bool shouldEmit = false;
        {
            std::lock_guard<std::mutex> lock(bestMutex);
            if (!hasBestMove || score > bestScore) {
                bestScore = score;
                bestMove = orderedMoves[index];
                hasBestMove = true;
            }
            const auto now = std::chrono::steady_clock::now();
            if (force || now - lastEmit >= std::chrono::milliseconds(250)) {
                lastEmit = now;
                info.depth = depth;
                info.scoreCp = std::clamp(bestScore, -MateScore, MateScore);
                info.nodes = nodes_.load();
                info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - searchStart).count());
                info.bestMove = bestMove;
                info.hasBestMove = hasBestMove;
                shouldEmit = hasBestMove;
            }
        }
        if (shouldEmit && infoCallback) {
            std::lock_guard<std::mutex> emitLock(emitMutex);
            infoCallback(info);
        }
    };

    const int workerCount = std::min<int>(std::max(1, threads_), static_cast<int>(orderedMoves.size()));
    if (workerCount <= 1) {
        for (std::size_t i = 0; i < orderedMoves.size(); ++i) {
            if (shouldStop()) {
                break;
            }
            Board next = board;
            applyMove(next, orderedMoves[i]);
            scores[i] = search(next, depth - 1, -MateScore, MateScore, rootSide);
            if (i < openingPenalties.size()) {
                scores[i] -= openingPenalties[i];
            }
            publish(i, scores[i], i + 1 == orderedMoves.size());
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
                scores[index] = search(next, depth - 1, -MateScore, MateScore, rootSide);
                if (index < openingPenalties.size()) {
                    scores[index] -= openingPenalties[index];
                }
                publish(index, scores[index], index + 1 == orderedMoves.size());
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

// 手の並べ替え (move ordering)
// αβ枝刈りの効率を上げるため、有望な手を先に探索する
// 駒取り > 王手 > 成り > 駒打ち の順に高いスコアを付与
std::vector<Move> LearningEngine::orderMoves(const Board& board, const std::vector<Move>& moves, Color rootSide) const {
    struct ScoredMove {
        Move move;
        int score = 0;
    };
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());
    for (const Move& move : moves) {
        scored.push_back(ScoredMove{move, moveOrderScore(board, move, rootSide)});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const ScoredMove& left, const ScoredMove& right) {
        return left.score > right.score;
    });
    std::vector<Move> ordered;
    ordered.reserve(scored.size());
    for (const ScoredMove& item : scored) {
        ordered.push_back(item.move);
    }
    return ordered;
}

// 各手の並べ替え優先度を計算
// MVV-LVA: 高い駒を安い駒で取るほど高スコア
int LearningEngine::moveOrderScore(const Board& board, const Move& move, Color rootSide) const {
    (void)rootSide;
    int score = 0;
    // 駒取り: 取る駒の価値が高く、取る側の駒が安いほど良い (MVV-LVA)
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
    // 王手になる手は優先度を上げる
    Board next = board;
    applyMove(next, move);
    if (isKingAttacked(next, next.side)) {
        score += 8000;
    }
    return score;
}

// 制限時間を超えたら探索を打ち切る
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

// Zobristハッシュによる差分更新済みの値を返す
// rootSideも混ぜて同じ局面でも探索視点が異なれば別エントリにする
std::uint64_t LearningEngine::boardHash(const Board& board, Color rootSide) const {
    return board.hash ^ (rootSide == Black ? 0x9E3779B97F4A7C15ULL : 0x6C62272E07BB0142ULL);
}

void LearningEngine::setLastSearchInfo(const SearchInfo& info) const {
    std::lock_guard<std::mutex> lock(lastSearchInfoMutex_);
    lastSearchInfo_ = info;
}

int LearningEngine::depthLimit() const {
    return searchDepth_ > 0 ? searchDepth_ : 128;
}

} // namespace shogi
