#include "nnue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

namespace shogi {

namespace nnue {

static int handTypeOffset(PieceType type) {
    switch (type) {
    case Pawn:   return 0;
    case Lance:  return 18;
    case Knight: return 22;
    case Silver: return 26;
    case Gold:   return 30;
    case Bishop: return 34;
    case Rook:   return 36;
    default:     return -1;
    }
}

int boardFeatureIndex(Color perspective, int pieceColor, PieceType type, int square) {
    bool isOwn = (pieceColor == static_cast<int>(perspective));
    int colorOffset = isOwn ? 0 : NumPieceTypes * BoardSize;
    int typeIdx = static_cast<int>(type) - 1;
    return colorOffset + typeIdx * BoardSize + square;
}

int handFeatureIndex(Color perspective, Color handColor, PieceType type, int count) {
    if (count <= 0) return -1;
    int offset = handTypeOffset(type);
    if (offset < 0) return -1;
    bool isOwn = (handColor == perspective);
    int base = BoardFeatures;
    int colorOffset = isOwn ? 0 : HandFeaturesPerColor;
    return base + colorOffset + offset + (count - 1);
}

} // namespace nnue

NNUENetwork::NNUENetwork() {
    initRandom();
}

void NNUENetwork::initRandom() {
    std::mt19937 rng(42);
    auto he = [&](int fanIn) {
        float scale = std::sqrt(2.0f / static_cast<float>(fanIn));
        std::normal_distribution<float> dist(0.0f, scale);
        return dist(rng);
    };

    for (int i = 0; i < nnue::InputDim; ++i)
        for (int j = 0; j < nnue::L0Size; ++j)
            l0Weights_[i][j] = he(nnue::InputDim);
    std::fill(std::begin(l0Biases_), std::end(l0Biases_), 0.0f);

    for (int i = 0; i < 2 * nnue::L0Size; ++i)
        for (int j = 0; j < nnue::L1Size; ++j)
            l1Weights_[i][j] = he(2 * nnue::L0Size);
    std::fill(std::begin(l1Biases_), std::end(l1Biases_), 0.0f);

    for (int i = 0; i < nnue::L1Size; ++i)
        for (int j = 0; j < nnue::L2Size; ++j)
            l2Weights_[i][j] = he(nnue::L1Size);
    std::fill(std::begin(l2Biases_), std::end(l2Biases_), 0.0f);

    for (int i = 0; i < nnue::L2Size; ++i)
        l3Weights_[i] = he(nnue::L2Size);
    l3Bias_ = 0.0f;
    loaded_ = false;
}

void NNUENetwork::computeAccumulator(const std::vector<int>& activeFeatures, float* output) const {
    std::copy(std::begin(l0Biases_), std::end(l0Biases_), output);
    for (int fi : activeFeatures) {
        if (fi < 0 || fi >= nnue::InputDim) continue;
        for (int j = 0; j < nnue::L0Size; ++j) {
            output[j] += l0Weights_[fi][j];
        }
    }
}

static std::vector<int> extractActiveFeatures(const Board& board, Color perspective) {
    std::vector<int> features;
    features.reserve(64);

    for (int sq = 0; sq < BoardSize; ++sq) {
        int piece = board.squares[sq];
        if (piece == 0) continue;
        PieceType type = typeOf(piece);
        int pieceColor = colorOf(piece);
        int fi = nnue::boardFeatureIndex(perspective, pieceColor, type, sq);
        features.push_back(fi);
    }

    for (PieceType pt : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        int bc = hand(board, Black)[pt];
        for (int c = 1; c <= bc; ++c) {
            int fi = nnue::handFeatureIndex(perspective, Black, pt, c);
            if (fi >= 0) features.push_back(fi);
        }
        int wc = hand(board, White)[pt];
        for (int c = 1; c <= wc; ++c) {
            int fi = nnue::handFeatureIndex(perspective, White, pt, c);
            if (fi >= 0) features.push_back(fi);
        }
    }

    return features;
}

nnue::FeatureDelta NNUENetwork::computeMoveDelta(const Board& board, const Move& move) const {
    nnue::FeatureDelta delta{};
    const Color mover = board.side;
    const int moverColor = static_cast<int>(mover);

    for (Color perspective : {Black, White}) {
        int* removed = perspective == Black ? delta.blackRemoved : delta.whiteRemoved;
        int* added = perspective == Black ? delta.blackAdded : delta.whiteAdded;
        int& nRemoved = perspective == Black ? delta.numBlackRemoved : delta.numWhiteRemoved;
        int& nAdded = perspective == Black ? delta.numBlackAdded : delta.numWhiteAdded;

        if (move.isDrop()) {
            added[nAdded++] = nnue::boardFeatureIndex(perspective, moverColor, move.drop, move.to);

            const int oldCount = hand(board, mover)[move.drop];
            int fi = nnue::handFeatureIndex(perspective, mover, move.drop, oldCount);
            if (fi >= 0) removed[nRemoved++] = fi;
            // newCount = oldCount - 1; if > 0, that feature stays active (cumulative)
        } else {
            removed[nRemoved++] = nnue::boardFeatureIndex(perspective, moverColor, move.piece, move.from);

            PieceType finalType = move.promote ? promote(move.piece) : move.piece;
            added[nAdded++] = nnue::boardFeatureIndex(perspective, moverColor, finalType, move.to);

            int captured = board.squares[move.to];
            if (captured != 0) {
                int capColor = colorOf(captured);
                PieceType capType = typeOf(captured);
                removed[nRemoved++] = nnue::boardFeatureIndex(perspective, capColor, capType, move.to);

                PieceType capBase = unpromote(capType);
                const int oldCount = hand(board, mover)[capBase];
                int newCount = oldCount + 1;
                added[nAdded++] = nnue::handFeatureIndex(perspective, mover, capBase, newCount);

                if (oldCount > 0) {
                    // nothing to remove: cumulative encoding means count=oldCount stays active
                }
            }
        }
    }

    return delta;
}

void NNUENetwork::computeAccumulatorFull(const Board& board, nnue::Accumulator& acc) const {
    auto blackFeatures = extractActiveFeatures(board, Black);
    auto whiteFeatures = extractActiveFeatures(board, White);
    computeAccumulator(blackFeatures, acc.black);
    computeAccumulator(whiteFeatures, acc.white);
    acc.computed = true;
}

void NNUENetwork::updateAccumulatorIncremental(const nnue::Accumulator& parent, const nnue::FeatureDelta& delta, nnue::Accumulator& child) const {
    std::copy(parent.black, parent.black + nnue::L0Size, child.black);
    for (int i = 0; i < delta.numBlackRemoved; ++i) {
        const int fi = delta.blackRemoved[i];
        if (fi >= 0 && fi < nnue::InputDim)
            for (int j = 0; j < nnue::L0Size; ++j) child.black[j] -= l0Weights_[fi][j];
    }
    for (int i = 0; i < delta.numBlackAdded; ++i) {
        const int fi = delta.blackAdded[i];
        if (fi >= 0 && fi < nnue::InputDim)
            for (int j = 0; j < nnue::L0Size; ++j) child.black[j] += l0Weights_[fi][j];
    }

    std::copy(parent.white, parent.white + nnue::L0Size, child.white);
    for (int i = 0; i < delta.numWhiteRemoved; ++i) {
        const int fi = delta.whiteRemoved[i];
        if (fi >= 0 && fi < nnue::InputDim)
            for (int j = 0; j < nnue::L0Size; ++j) child.white[j] -= l0Weights_[fi][j];
    }
    for (int i = 0; i < delta.numWhiteAdded; ++i) {
        const int fi = delta.whiteAdded[i];
        if (fi >= 0 && fi < nnue::InputDim)
            for (int j = 0; j < nnue::L0Size; ++j) child.white[j] += l0Weights_[fi][j];
    }

    child.computed = true;
}

int NNUENetwork::evaluateFromAccumulator(const nnue::Accumulator& acc, Color perspective) const {
    float clampedBlack[nnue::L0Size], clampedWhite[nnue::L0Size];
    for (int i = 0; i < nnue::L0Size; ++i) {
        clampedBlack[i] = std::clamp(acc.black[i], 0.0f, 1.0f);
        clampedWhite[i] = std::clamp(acc.white[i], 0.0f, 1.0f);
    }

    float concat[2 * nnue::L0Size];
    if (perspective == Black) {
        std::copy(clampedBlack, clampedBlack + nnue::L0Size, concat);
        std::copy(clampedWhite, clampedWhite + nnue::L0Size, concat + nnue::L0Size);
    } else {
        std::copy(clampedWhite, clampedWhite + nnue::L0Size, concat);
        std::copy(clampedBlack, clampedBlack + nnue::L0Size, concat + nnue::L0Size);
    }

    float h1[nnue::L1Size];
    for (int j = 0; j < nnue::L1Size; ++j) {
        float sum = l1Biases_[j];
        for (int i = 0; i < 2 * nnue::L0Size; ++i) sum += concat[i] * l1Weights_[i][j];
        h1[j] = std::clamp(sum, 0.0f, 1.0f);
    }

    float h2[nnue::L2Size];
    for (int j = 0; j < nnue::L2Size; ++j) {
        float sum = l2Biases_[j];
        for (int i = 0; i < nnue::L1Size; ++i) sum += h1[i] * l2Weights_[i][j];
        h2[j] = std::clamp(sum, 0.0f, 1.0f);
    }

    float output = l3Bias_;
    for (int i = 0; i < nnue::L2Size; ++i) output += h2[i] * l3Weights_[i];

    return static_cast<int>(output * 600.0f);
}

int NNUENetwork::evaluate(const Board& board, Color perspective) const {
    auto blackFeatures = extractActiveFeatures(board, Black);
    auto whiteFeatures = extractActiveFeatures(board, White);

    float accBlack[nnue::L0Size], accWhite[nnue::L0Size];
    computeAccumulator(blackFeatures, accBlack);
    computeAccumulator(whiteFeatures, accWhite);

    for (int i = 0; i < nnue::L0Size; ++i) {
        accBlack[i] = std::clamp(accBlack[i], 0.0f, 1.0f);
        accWhite[i] = std::clamp(accWhite[i], 0.0f, 1.0f);
    }

    float concat[2 * nnue::L0Size];
    if (perspective == Black) {
        std::copy(accBlack, accBlack + nnue::L0Size, concat);
        std::copy(accWhite, accWhite + nnue::L0Size, concat + nnue::L0Size);
    } else {
        std::copy(accWhite, accWhite + nnue::L0Size, concat);
        std::copy(accBlack, accBlack + nnue::L0Size, concat + nnue::L0Size);
    }

    // L1
    float h1[nnue::L1Size];
    for (int j = 0; j < nnue::L1Size; ++j) {
        float sum = l1Biases_[j];
        for (int i = 0; i < 2 * nnue::L0Size; ++i) {
            sum += concat[i] * l1Weights_[i][j];
        }
        h1[j] = std::clamp(sum, 0.0f, 1.0f);
    }

    // L2
    float h2[nnue::L2Size];
    for (int j = 0; j < nnue::L2Size; ++j) {
        float sum = l2Biases_[j];
        for (int i = 0; i < nnue::L1Size; ++i) {
            sum += h1[i] * l2Weights_[i][j];
        }
        h2[j] = std::clamp(sum, 0.0f, 1.0f);
    }

    // L3
    float output = l3Bias_;
    for (int i = 0; i < nnue::L2Size; ++i) {
        output += h2[i] * l3Weights_[i];
    }

    return static_cast<int>(output * 600.0f);
}

bool NNUENetwork::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "NNUE", 4) != 0) return false;

    file.read(reinterpret_cast<char*>(l0Weights_), sizeof(l0Weights_));
    file.read(reinterpret_cast<char*>(l0Biases_), sizeof(l0Biases_));
    file.read(reinterpret_cast<char*>(l1Weights_), sizeof(l1Weights_));
    file.read(reinterpret_cast<char*>(l1Biases_), sizeof(l1Biases_));
    file.read(reinterpret_cast<char*>(l2Weights_), sizeof(l2Weights_));
    file.read(reinterpret_cast<char*>(l2Biases_), sizeof(l2Biases_));
    file.read(reinterpret_cast<char*>(l3Weights_), sizeof(l3Weights_));
    file.read(reinterpret_cast<char*>(&l3Bias_), sizeof(l3Bias_));

    if (!file) return false;
    loaded_ = true;
    return true;
}

bool NNUENetwork::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    file.write("NNUE", 4);
    file.write(reinterpret_cast<const char*>(l0Weights_), sizeof(l0Weights_));
    file.write(reinterpret_cast<const char*>(l0Biases_), sizeof(l0Biases_));
    file.write(reinterpret_cast<const char*>(l1Weights_), sizeof(l1Weights_));
    file.write(reinterpret_cast<const char*>(l1Biases_), sizeof(l1Biases_));
    file.write(reinterpret_cast<const char*>(l2Weights_), sizeof(l2Weights_));
    file.write(reinterpret_cast<const char*>(l2Biases_), sizeof(l2Biases_));
    file.write(reinterpret_cast<const char*>(l3Weights_), sizeof(l3Weights_));
    file.write(reinterpret_cast<const char*>(&l3Bias_), sizeof(l3Bias_));

    return file.good();
}

} // namespace shogi
