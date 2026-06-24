#pragma once

#include "shogi_types.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace shogi {

namespace nnue {

constexpr int NumPieceTypes = 14;
constexpr int BoardFeatures = 2 * NumPieceTypes * BoardSize; // 2268
constexpr int HandMaxCounts[] = {0, 18, 4, 4, 4, 4, 2, 2}; // indexed by PieceType 1-7
constexpr int HandFeaturesPerColor = 18 + 4 + 4 + 4 + 4 + 2 + 2; // 38
constexpr int HandFeatures = 2 * HandFeaturesPerColor; // 76
constexpr int InputDim = BoardFeatures + HandFeatures; // 2344

constexpr int L0Size = 256;
constexpr int L1Size = 32;
constexpr int L2Size = 32;

int boardFeatureIndex(Color perspective, int pieceColor, PieceType type, int square);
int handFeatureIndex(Color perspective, Color handColor, PieceType type, int count);

struct Accumulator {
    alignas(64) float black[L0Size];
    alignas(64) float white[L0Size];
    bool computed = false;
};

struct FeatureDelta {
    int blackRemoved[4], blackAdded[4];
    int whiteRemoved[4], whiteAdded[4];
    int numBlackRemoved = 0, numBlackAdded = 0;
    int numWhiteRemoved = 0, numWhiteAdded = 0;
};

} // namespace nnue

class NNUENetwork {
public:
    NNUENetwork();

    int evaluate(const Board& board, Color perspective) const;
    bool load(const std::string& path);
    bool save(const std::string& path) const;
    bool loaded() const { return loaded_; }
    void initRandom();

    nnue::FeatureDelta computeMoveDelta(const Board& board, const Move& move) const;
    void computeAccumulatorFull(const Board& board, nnue::Accumulator& acc) const;
    void updateAccumulatorIncremental(const nnue::Accumulator& parent, const nnue::FeatureDelta& delta, nnue::Accumulator& child) const;
    int evaluateFromAccumulator(const nnue::Accumulator& acc, Color perspective) const;

private:
    void computeAccumulator(const std::vector<int>& activeFeatures, float* output) const;

    float l0Weights_[nnue::InputDim][nnue::L0Size]{};
    float l0Biases_[nnue::L0Size]{};
    float l1Weights_[2 * nnue::L0Size][nnue::L1Size]{};
    float l1Biases_[nnue::L1Size]{};
    float l2Weights_[nnue::L1Size][nnue::L2Size]{};
    float l2Biases_[nnue::L2Size]{};
    float l3Weights_[nnue::L2Size]{};
    float l3Bias_ = 0.0f;
    bool loaded_ = false;
};

} // namespace shogi
