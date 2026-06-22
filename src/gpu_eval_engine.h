#pragma once

#include "evaluation.h"
#include "gpu_bridge.h"
#include "search_types.h"
#include "shogi_types.h"

#include <mutex>
#include <random>
#include <string>

namespace shogi {

class GpuEvalEngine {
public:
    GpuEvalEngine();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback);

    void setMaxMoveTimeMs(int milliseconds);
    void setOpeningSafety(bool enabled);
    void setGpuPython(const std::string& python);
    void setGpuScript(const std::string& script);
    void setGpuModel(const std::string& model);
    void setGpuDevice(const std::string& device);
    SearchInfo lastSearchInfo() const;

private:
    Evaluator evaluator_;
    GpuBridge gpu_;
    mutable SearchInfo lastSearchInfo_{};
    mutable std::mutex infoMutex_;
    int maxMoveTimeMs_ = 3000;
    bool openingSafety_ = true;
    std::mt19937 rng_;
};

} // namespace shogi
