#pragma once

#include "shogi_types.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace shogi {

namespace nnue {

constexpr int NumPieceTypes = 14;
constexpr int NumNonKingTypes = 13; // King excluded from HalfKP features
constexpr int NumColoredPieceTypes = 2 * NumNonKingTypes; // own 13 + opponent 13 = 26

// HalfKP: kingSquare(81) × coloredPieceType(26) × pieceSquare(81)
constexpr int BoardFeatures = BoardSize * NumColoredPieceTypes * BoardSize; // 81 * 26 * 81 = 170586

constexpr int HandMaxCounts[] = {0, 18, 4, 4, 4, 4, 2, 2}; // indexed by PieceType 1-7
constexpr int HandFeaturesPerColor = 18 + 4 + 4 + 4 + 4 + 2 + 2; // 38
constexpr int HandFeatures = 2 * HandFeaturesPerColor; // 76
constexpr int InputDim = BoardFeatures + HandFeatures; // 170662

constexpr int L0Size = 512;
constexpr int L1Size = 32;
constexpr int L2Size = 32;
constexpr int WeightScale = 64;

int boardFeatureIndex(int kingSquare, bool isOwnPiece, PieceType type, int pieceSquare);
int handFeatureIndex(bool isOwnHand, PieceType type, int count);

struct Accumulator {
    alignas(64) std::int32_t black[L0Size];
    alignas(64) std::int32_t white[L0Size];
    bool computed = false;
    int blackKingSq = -1;
    int whiteKingSq = -1;
};

struct FeatureDelta {
    int blackRemoved[6], blackAdded[6];
    int whiteRemoved[6], whiteAdded[6];
    int numBlackRemoved = 0, numBlackAdded = 0;
    int numWhiteRemoved = 0, numWhiteAdded = 0;
    bool blackNeedsFullRecompute = false;
    bool whiteNeedsFullRecompute = false;
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
    void updateAccumulatorIncremental(const Board& boardAfterMove, const nnue::Accumulator& parent, const nnue::FeatureDelta& delta, nnue::Accumulator& child) const;
    int evaluateFromAccumulator(const nnue::Accumulator& acc, Color perspective) const;

private:
    void computeAccumulator(const std::vector<int>& activeFeatures, std::int32_t* output) const;

    std::vector<std::int16_t> l0Weights_; // [InputDim][L0Size] flattened
    std::int32_t l0Biases_[nnue::L0Size]{};
    float l1Weights_[2 * nnue::L0Size][nnue::L1Size]{};
    float l1Biases_[nnue::L1Size]{};
    float l2Weights_[nnue::L1Size][nnue::L2Size]{};
    float l2Biases_[nnue::L2Size]{};
    float l3Weights_[nnue::L2Size]{};
    float l3Bias_ = 0.0f;
    bool loaded_ = false;
};

} // namespace shogi
