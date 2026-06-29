#pragma once

#include "shogi_types.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace shogi {

constexpr int PolicySize = 2187; // 81 squares * 27 channels

struct NNOutput {
    double value = 0.0;
    std::vector<double> policy;
};

struct NNBridgeSettings {
    bool enabled = false;
    std::string model = "nn_model.onnx";
    std::string device = "auto";
};

class OnnxInference;

class NNBridge {
public:
    NNBridge();
    ~NNBridge();

    void setEnabled(bool enabled);
    bool enabled() const;
    void setModel(const std::string& model);
    void setDevice(const std::string& device);

    NNOutput evaluate(const Board& board);
    std::vector<NNOutput> evaluateBatch(const std::vector<Board>& boards);

    static int moveToIndex(const Move& move, Color side);

    bool ensureReady();
    bool isReady() const;
    void shutdown();
    const std::string& lastError() const { return lastError_; }
    const std::string& modelPath() const { return settings_.model; }
    std::string deviceUsed() const;
    std::string cudaError() const;

private:
    NNOutput makeFallbackOutput() const;

    std::unique_ptr<OnnxInference> onnx_;
    bool onnxAttempted_ = false;

    NNBridgeSettings settings_;
    mutable std::mutex mutex_;
    std::string lastError_;
};

} // namespace shogi
