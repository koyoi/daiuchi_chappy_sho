#include "nnue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#if defined(__AVX2__)
#define NNUE_USE_AVX2 1
#include <immintrin.h>
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
// MSVC x64: SSE4.1 always available; AVX2 when /arch:AVX2 is set
#include <intrin.h>
#include <immintrin.h>
// Runtime AVX2 check not needed since we build with /arch:AVX2 in Release
#define NNUE_USE_AVX2 1
#elif defined(__SSE4_1__)
#define NNUE_USE_SSE 1
#include <smmintrin.h>
#endif

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

static int pieceTypeIndex(bool isOwn, PieceType type) {
    // King is excluded from HalfKP board features
    if (type == King) return -1;
    int typeIdx = static_cast<int>(type) - 1; // 0..13, but King(8)->7 excluded
    if (typeIdx >= 7) typeIdx -= 1; // shift promoted pieces down (skip King slot)
    // typeIdx now: Pawn=0,Lance=1,Knight=2,Silver=3,Gold=4,Bishop=5,Rook=6,
    //   ProPawn=7,ProLance=8,ProKnight=9,ProSilver=10,Horse=11,Dragon=12
    int colorOffset = isOwn ? 0 : NumNonKingTypes;
    return colorOffset + typeIdx;
}

int boardFeatureIndex(int kingSquare, bool isOwnPiece, PieceType type, int pieceSquare) {
    int ptIdx = pieceTypeIndex(isOwnPiece, type);
    if (ptIdx < 0) return -1;
    return kingSquare * NumColoredPieceTypes * BoardSize + ptIdx * BoardSize + pieceSquare;
}

int handFeatureIndex(bool isOwnHand, PieceType type, int count) {
    if (count <= 0) return -1;
    int offset = handTypeOffset(type);
    if (offset < 0) return -1;
    int colorOffset = isOwnHand ? 0 : HandFeaturesPerColor;
    return BoardFeatures + colorOffset + offset + (count - 1);
}

} // namespace nnue

NNUENetwork::NNUENetwork() {
    l0Weights_.resize(static_cast<std::size_t>(nnue::InputDim) * nnue::L0Size, 0);
    initRandom();
}

void NNUENetwork::initRandom() {
    std::mt19937 rng(42);
    auto he = [&](int fanIn) {
        float scale = std::sqrt(2.0f / static_cast<float>(fanIn));
        std::normal_distribution<float> dist(0.0f, scale);
        return dist(rng);
    };

    l0Weights_.resize(static_cast<std::size_t>(nnue::InputDim) * nnue::L0Size);
    for (std::size_t i = 0; i < l0Weights_.size(); ++i)
        l0Weights_[i] = static_cast<std::int16_t>(std::round(he(nnue::InputDim) * nnue::WeightScale));
    for (int j = 0; j < nnue::L0Size; ++j)
        l0Biases_[j] = 0;

    {
        float scale = std::sqrt(1.0f / static_cast<float>(2 * nnue::L0Size));
        std::normal_distribution<float> dist(0.0f, scale);
        for (int i = 0; i < 2 * nnue::L0Size; ++i)
            outWeights_[i] = dist(rng);
    }
    outBias_ = 0.0f;
    loaded_ = false;
}

namespace {

inline void accAddWeights(std::int32_t* acc, const std::int16_t* weights, int size) {
#if NNUE_USE_AVX2
    // Process 8 int32 at a time using AVX2
    // Expand int16 weights to int32 and add
    for (int j = 0; j < size; j += 8) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j));
        __m128i w128 = _mm_load_si128(reinterpret_cast<const __m128i*>(weights + j));
        __m256i w = _mm256_cvtepi16_epi32(w128);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j), _mm256_add_epi32(a, w));
    }
#elif NNUE_USE_SSE
    for (int j = 0; j < size; j += 4) {
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(acc + j));
        __m128i w = _mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + j)));
        _mm_store_si128(reinterpret_cast<__m128i*>(acc + j), _mm_add_epi32(a, w));
    }
#else
    for (int j = 0; j < size; ++j) acc[j] += weights[j];
#endif
}

inline void accSubWeights(std::int32_t* acc, const std::int16_t* weights, int size) {
#if NNUE_USE_AVX2
    for (int j = 0; j < size; j += 8) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j));
        __m128i w128 = _mm_load_si128(reinterpret_cast<const __m128i*>(weights + j));
        __m256i w = _mm256_cvtepi16_epi32(w128);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j), _mm256_sub_epi32(a, w));
    }
#elif NNUE_USE_SSE
    for (int j = 0; j < size; j += 4) {
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(acc + j));
        __m128i w = _mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + j)));
        _mm_store_si128(reinterpret_cast<__m128i*>(acc + j), _mm_sub_epi32(a, w));
    }
#else
    for (int j = 0; j < size; ++j) acc[j] -= weights[j];
#endif
}

} // namespace

void NNUENetwork::computeAccumulator(const std::vector<int>& activeFeatures, std::int32_t* output) const {
    std::copy(std::begin(l0Biases_), std::end(l0Biases_), output);
    for (int fi : activeFeatures) {
        if (fi < 0 || fi >= nnue::InputDim) continue;
        const std::int16_t* w = &l0Weights_[static_cast<std::size_t>(fi) * nnue::L0Size];
        accAddWeights(output, w, nnue::L0Size);
    }
}

static std::vector<int> extractActiveFeatures(const Board& board, Color perspective) {
    std::vector<int> features;
    features.reserve(64);

    const int kingSquare = (perspective == Black) ? board.blackKingSquare : board.whiteKingSquare;
    if (kingSquare < 0) return features;

    for (int sq = 0; sq < BoardSize; ++sq) {
        int piece = board.squares[sq];
        if (piece == 0) continue;
        PieceType type = typeOf(piece);
        if (type == King) continue;
        int pieceColor = colorOf(piece);
        bool isOwn = (pieceColor == static_cast<int>(perspective));
        int fi = nnue::boardFeatureIndex(kingSquare, isOwn, type, sq);
        if (fi >= 0) features.push_back(fi);
    }

    for (PieceType pt : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        int bc = hand(board, Black)[pt];
        for (int c = 1; c <= bc; ++c) {
            bool isOwn = (perspective == Black);
            int fi = nnue::handFeatureIndex(isOwn, pt, c);
            if (fi >= 0) features.push_back(fi);
        }
        int wc = hand(board, White)[pt];
        for (int c = 1; c <= wc; ++c) {
            bool isOwn = (perspective == White);
            int fi = nnue::handFeatureIndex(isOwn, pt, c);
            if (fi >= 0) features.push_back(fi);
        }
    }

    return features;
}

nnue::FeatureDelta NNUENetwork::computeMoveDelta(const Board& board, const Move& move) const {
    nnue::FeatureDelta delta{};
    const Color mover = board.side;
    const int moverColor = static_cast<int>(mover);

    const int blackKingSq = board.blackKingSquare;
    const int whiteKingSq = board.whiteKingSquare;

    bool movingKing = false;
    if (!move.isDrop() && typeOf(board.squares[move.from]) == King) {
        movingKing = true;
    }

    for (Color perspective : {Black, White}) {
        int* removed = perspective == Black ? delta.blackRemoved : delta.whiteRemoved;
        int* added = perspective == Black ? delta.blackAdded : delta.whiteAdded;
        int& nRemoved = perspective == Black ? delta.numBlackRemoved : delta.numWhiteRemoved;
        int& nAdded = perspective == Black ? delta.numBlackAdded : delta.numWhiteAdded;
        bool& needsRecompute = perspective == Black ? delta.blackNeedsFullRecompute : delta.whiteNeedsFullRecompute;

        if (movingKing && mover == perspective) {
            needsRecompute = true;
            continue;
        }

        const int kingSq = (perspective == Black) ? blackKingSq : whiteKingSq;
        if (kingSq < 0) { needsRecompute = true; continue; }

        if (move.isDrop()) {
            PieceType dropType = move.drop;
            if (dropType != King) {
                bool isOwn = (moverColor == static_cast<int>(perspective));
                int fi = nnue::boardFeatureIndex(kingSq, isOwn, dropType, move.to);
                if (fi >= 0) added[nAdded++] = fi;
            }

            const int oldCount = hand(board, mover)[move.drop];
            int fi = nnue::handFeatureIndex(mover == perspective, move.drop, oldCount);
            if (fi >= 0) removed[nRemoved++] = fi;
        } else {
            PieceType movingPiece = move.piece;
            if (movingPiece != King) {
                bool isOwn = (moverColor == static_cast<int>(perspective));
                int fi = nnue::boardFeatureIndex(kingSq, isOwn, movingPiece, move.from);
                if (fi >= 0) removed[nRemoved++] = fi;

                PieceType finalType = move.promote ? promote(movingPiece) : movingPiece;
                fi = nnue::boardFeatureIndex(kingSq, isOwn, finalType, move.to);
                if (fi >= 0) added[nAdded++] = fi;
            }

            int captured = board.squares[move.to];
            if (captured != 0) {
                int capColor = colorOf(captured);
                PieceType capType = typeOf(captured);
                if (capType != King) {
                    bool capIsOwn = (capColor == static_cast<int>(perspective));
                    int fi = nnue::boardFeatureIndex(kingSq, capIsOwn, capType, move.to);
                    if (fi >= 0) removed[nRemoved++] = fi;
                }

                PieceType capBase = unpromote(capType);
                const int oldCount = hand(board, mover)[capBase];
                int newCount = oldCount + 1;
                int fi = nnue::handFeatureIndex(mover == perspective, capBase, newCount);
                if (fi >= 0) added[nAdded++] = fi;
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
    acc.blackKingSq = board.blackKingSquare;
    acc.whiteKingSq = board.whiteKingSquare;
    acc.computed = true;
}

void NNUENetwork::updateAccumulatorIncremental(const Board& boardAfterMove, const nnue::Accumulator& parent, const nnue::FeatureDelta& delta, nnue::Accumulator& child) const {
    if (delta.blackNeedsFullRecompute) {
        auto features = extractActiveFeatures(boardAfterMove, Black);
        computeAccumulator(features, child.black);
        child.blackKingSq = boardAfterMove.blackKingSquare;
    } else {
        std::copy(parent.black, parent.black + nnue::L0Size, child.black);
        for (int i = 0; i < delta.numBlackRemoved; ++i) {
            const int fi = delta.blackRemoved[i];
            if (fi >= 0 && fi < nnue::InputDim)
                accSubWeights(child.black, &l0Weights_[static_cast<std::size_t>(fi) * nnue::L0Size], nnue::L0Size);
        }
        for (int i = 0; i < delta.numBlackAdded; ++i) {
            const int fi = delta.blackAdded[i];
            if (fi >= 0 && fi < nnue::InputDim)
                accAddWeights(child.black, &l0Weights_[static_cast<std::size_t>(fi) * nnue::L0Size], nnue::L0Size);
        }
        child.blackKingSq = parent.blackKingSq;
    }

    if (delta.whiteNeedsFullRecompute) {
        auto features = extractActiveFeatures(boardAfterMove, White);
        computeAccumulator(features, child.white);
        child.whiteKingSq = boardAfterMove.whiteKingSquare;
    } else {
        std::copy(parent.white, parent.white + nnue::L0Size, child.white);
        for (int i = 0; i < delta.numWhiteRemoved; ++i) {
            const int fi = delta.whiteRemoved[i];
            if (fi >= 0 && fi < nnue::InputDim)
                accSubWeights(child.white, &l0Weights_[static_cast<std::size_t>(fi) * nnue::L0Size], nnue::L0Size);
        }
        for (int i = 0; i < delta.numWhiteAdded; ++i) {
            const int fi = delta.whiteAdded[i];
            if (fi >= 0 && fi < nnue::InputDim)
                accAddWeights(child.white, &l0Weights_[static_cast<std::size_t>(fi) * nnue::L0Size], nnue::L0Size);
        }
        child.whiteKingSq = parent.whiteKingSq;
    }

    child.computed = true;
}

int NNUENetwork::evaluateFromAccumulator(const nnue::Accumulator& acc, Color perspective) const {
    constexpr float invScale = 1.0f / nnue::WeightScale;

    const std::int32_t* own = (perspective == Black) ? acc.black : acc.white;
    const std::int32_t* opp = (perspective == Black) ? acc.white : acc.black;

    float output = outBias_;

#if NNUE_USE_AVX2
    const __m256 vInvScale = _mm256_set1_ps(invScale);
    const __m256 vZero = _mm256_setzero_ps();
    const __m256 vOne = _mm256_set1_ps(1.0f);
    __m256 vSum = _mm256_setzero_ps();
    for (int i = 0; i < nnue::L0Size; i += 8) {
        __m256i vi = _mm256_load_si256(reinterpret_cast<const __m256i*>(own + i));
        __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), vInvScale);
        vf = _mm256_min_ps(_mm256_max_ps(vf, vZero), vOne);
        vf = _mm256_mul_ps(vf, vf);
        __m256 vw = _mm256_loadu_ps(outWeights_ + i);
        vSum = _mm256_add_ps(vSum, _mm256_mul_ps(vf, vw));
    }
    for (int i = 0; i < nnue::L0Size; i += 8) {
        __m256i vi = _mm256_load_si256(reinterpret_cast<const __m256i*>(opp + i));
        __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), vInvScale);
        vf = _mm256_min_ps(_mm256_max_ps(vf, vZero), vOne);
        vf = _mm256_mul_ps(vf, vf);
        __m256 vw = _mm256_loadu_ps(outWeights_ + nnue::L0Size + i);
        vSum = _mm256_add_ps(vSum, _mm256_mul_ps(vf, vw));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vSum);
    for (int i = 0; i < 8; ++i) output += tmp[i];
#else
    auto screlu = [](float x) { float c = std::clamp(x, 0.0f, 1.0f); return c * c; };
    for (int i = 0; i < nnue::L0Size; ++i)
        output += screlu(static_cast<float>(own[i]) * invScale) * outWeights_[i];
    for (int i = 0; i < nnue::L0Size; ++i)
        output += screlu(static_cast<float>(opp[i]) * invScale) * outWeights_[nnue::L0Size + i];
#endif

    return static_cast<int>(output * 600.0f);
}

int NNUENetwork::evaluate(const Board& board, Color perspective) const {
    nnue::Accumulator acc;
    computeAccumulatorFull(board, acc);
    return evaluateFromAccumulator(acc, perspective);
}

bool NNUENetwork::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    char magic[4];
    file.read(magic, 4);

    if (std::strncmp(magic, "NNU5", 4) == 0) {
        std::int32_t storedL0Size = 0;
        file.read(reinterpret_cast<char*>(&storedL0Size), sizeof(storedL0Size));
        if (storedL0Size != nnue::L0Size) return false;
        const std::size_t l0WeightSize = static_cast<std::size_t>(nnue::InputDim) * nnue::L0Size;
        l0Weights_.resize(l0WeightSize);
        file.read(reinterpret_cast<char*>(l0Weights_.data()), l0WeightSize * sizeof(std::int16_t));
        file.read(reinterpret_cast<char*>(l0Biases_), sizeof(l0Biases_));
        file.read(reinterpret_cast<char*>(outWeights_), sizeof(outWeights_));
        file.read(reinterpret_cast<char*>(&outBias_), sizeof(outBias_));
        if (!file) return false;
        loaded_ = true;
        return true;
    }

    return false;
}

bool NNUENetwork::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    file.write("NNU5", 4);
    const std::int32_t storedL0Size = nnue::L0Size;
    file.write(reinterpret_cast<const char*>(&storedL0Size), sizeof(storedL0Size));
    const std::size_t l0WeightSize = static_cast<std::size_t>(nnue::InputDim) * nnue::L0Size;
    file.write(reinterpret_cast<const char*>(l0Weights_.data()), l0WeightSize * sizeof(std::int16_t));
    file.write(reinterpret_cast<const char*>(l0Biases_), sizeof(l0Biases_));
    file.write(reinterpret_cast<const char*>(outWeights_), sizeof(outWeights_));
    file.write(reinterpret_cast<const char*>(&outBias_), sizeof(outBias_));

    return file.good();
}

} // namespace shogi
