#pragma once

#include "shogi_types.h"

#include <array>
#include <mutex>
#include <string>
#include <vector>

#ifdef HAS_ONNXRUNTIME
#include <memory>
#endif

namespace shogi {

constexpr int PolicySize = 2187; // 81 squares * 27 channels

struct NNOutput {
    double value = 0.0;
    std::vector<double> policy;
};

struct NNBridgeSettings {
    bool enabled = false;
#ifdef _WIN32
    std::string python = "..\\..\\.venv\\Scripts\\python.exe";
#else
    std::string python = "python";
#endif
    std::string script = "tools/nn_eval.py";
#ifdef HAS_ONNXRUNTIME
    std::string model = "nn_model.onnx";
#else
    std::string model = "nn_model.pt";
#endif
    std::string device = "auto";
};

#ifdef HAS_ONNXRUNTIME
class OnnxInference;
#endif

class NNBridge {
public:
    NNBridge();
    ~NNBridge();

    void setEnabled(bool enabled);
    bool enabled() const;
    void setPython(const std::string& python);
    void setScript(const std::string& script);
    void setModel(const std::string& model);
    void setDevice(const std::string& device);

    NNOutput evaluate(const Board& board);
    std::vector<NNOutput> evaluateBatch(const std::vector<Board>& boards);
    bool train(const std::string& dataPath, int epochs = 10);

    static int moveToIndex(const Move& move, Color side);
    static std::string encodeBoardState(const Board& board);

    bool ensureProcess();
    void shutdown();
    const std::string& lastError() const { return lastError_; }
    const std::string& modelPath() const { return settings_.model; }

private:
    bool sendLine(const std::string& line);
    std::string recvLine();
    std::string recvLine(int timeoutMs);
    NNOutput makeFallbackOutput() const;

#ifdef HAS_ONNXRUNTIME
    bool tryOnnx();
    std::unique_ptr<OnnxInference> onnx_;
    bool onnxAttempted_ = false;
#endif

    NNBridgeSettings settings_;
    mutable std::mutex mutex_;

#ifdef _WIN32
    void* childStdinWrite_ = nullptr;
    void* childStdoutRead_ = nullptr;
    void* processHandle_ = nullptr;
#else
    int childStdinFd_ = -1;
    int childStdoutFd_ = -1;
    int childPid_ = -1;
#endif
    bool processRunning_ = false;
    std::string lastError_;
};

} // namespace shogi
