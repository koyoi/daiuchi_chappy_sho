#pragma once

#include "shogi_types.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace shogi {

constexpr int AlphaPolicySize = 2187;
constexpr int AlphaInputChannels = 45;

struct AlphaNNOutput {
    double value = 0.0;
    std::array<double, 3> wdl = {0.0, 0.0, 0.0};
    std::vector<double> policy;
};

class AlphaOnnxInference {
public:
    AlphaOnnxInference();
    ~AlphaOnnxInference();

    bool loadModel(const std::string& modelPath, const std::string& device = "auto");
    bool isLoaded() const;
    const std::string& lastError() const { return lastError_; }
    const std::string& modelPath() const { return modelPath_; }

    AlphaNNOutput evaluate(const Board& board);
    std::vector<AlphaNNOutput> evaluateBatch(const std::vector<Board>& boards);

    static int moveToIndex(const Move& move, Color side);

private:
    void encodeBoardSpatial(const Board& board, int batchIdx,
                            std::vector<float>& features) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string lastError_;
    std::string modelPath_;
};

} // namespace shogi
