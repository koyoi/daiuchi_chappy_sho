#pragma once

#include "nn_bridge.h"
#include "shogi_types.h"

#include <memory>
#include <string>
#include <vector>

namespace shogi {

class OnnxInference {
public:
    OnnxInference();
    ~OnnxInference();

    bool loadModel(const std::string& modelPath, const std::string& device = "auto");
    bool isLoaded() const;
    const std::string& lastError() const { return lastError_; }

    NNOutput evaluate(const Board& board);
    std::vector<NNOutput> evaluateBatch(const std::vector<Board>& boards);

private:
    void encodeBoardDirect(const Board& board, int batchIdx,
                           std::vector<int64_t>& boardData,
                           std::vector<int64_t>& handData,
                           std::vector<int64_t>& sideData,
                           std::vector<int64_t>& ownAtkData,
                           std::vector<int64_t>& oppAtkData) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string lastError_;
};

} // namespace shogi
