#pragma once

#include "evaluation.h"

#include <string>
#include <vector>

namespace shogi {

struct GpuBridgeSettings {
    bool enabled = false;
    bool trainOnGameEnd = false;
#ifdef _WIN32
    std::string python = "..\\..\\.venv\\Scripts\\python.exe";
#else
    std::string python = "python";
#endif
    std::string script = "tools/gpu_eval.py";
    std::string model = "gpu_model.pt";
    std::string device = "auto";
};

class GpuBridge {
public:
    void setEnabled(bool enabled);
    bool enabled() const;
    void setTrainOnGameEnd(bool enabled);
    bool trainOnGameEnd() const;
    void setPython(const std::string& python);
    void setScript(const std::string& script);
    void setModel(const std::string& model);
    void setDevice(const std::string& device);

    bool score(const std::vector<FeatureVector>& features, std::vector<int>& scores) const;
    bool train(const std::string& trainingDataPath) const;

private:
    GpuBridgeSettings settings_;
};

} // namespace shogi
