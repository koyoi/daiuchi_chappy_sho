#include "gpu_bridge.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace shogi {

namespace {

std::string quoteArg(const std::string& value) {
    std::string quoted = "\"";
    for (char c : value) {
        if (c == '"') {
            quoted += "\\\"";
        } else {
            quoted += c;
        }
    }
    quoted += "\"";
    return quoted;
}

std::filesystem::path tempPath(const std::string& suffix) {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("kishi_to_gpu_" + std::to_string(tick) + suffix);
}

std::string nullRedirect() {
#ifdef _WIN32
    return " > NUL 2> NUL";
#else
    return " > /dev/null 2> /dev/null";
#endif
}

std::string appendLogRedirect(const std::string& path) {
    return " >> " + quoteArg(path) + " 2>&1";
}

std::string buildCommand(const std::string& executable, const std::string& arguments, const std::string& redirect) {
    const bool needsQuotedExecutable = executable.find_first_of(" \t\"") != std::string::npos;
#ifdef _WIN32
    if (needsQuotedExecutable) {
        return "\"" + quoteArg(executable) + arguments + "\"" + redirect;
    }
#endif
    return (needsQuotedExecutable ? quoteArg(executable) : executable) + arguments + redirect;
}

bool writeFeatureFile(const std::filesystem::path& path, const std::vector<FeatureVector>& features) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    for (const FeatureVector& row : features) {
        for (int i = 0; i < FeatureCount; ++i) {
            if (i > 0) {
                output << '\t';
            }
            output << row[i];
        }
        output << '\n';
    }
    return true;
}

bool readScores(const std::filesystem::path& path, std::vector<int>& scores, std::size_t expectedCount) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    scores.clear();
    double value = 0.0;
    while (input >> value) {
        scores.push_back(static_cast<int>(value));
    }
    return scores.size() == expectedCount;
}

} // namespace

void GpuBridge::setEnabled(bool enabled) {
    settings_.enabled = enabled;
}

bool GpuBridge::enabled() const {
    return settings_.enabled;
}

void GpuBridge::setTrainOnGameEnd(bool enabled) {
    settings_.trainOnGameEnd = enabled;
}

bool GpuBridge::trainOnGameEnd() const {
    return settings_.trainOnGameEnd;
}

void GpuBridge::setPython(const std::string& python) {
    if (!python.empty()) {
        settings_.python = python;
    }
}

void GpuBridge::setScript(const std::string& script) {
    if (!script.empty()) {
        settings_.script = script;
    }
}

void GpuBridge::setModel(const std::string& model) {
    if (!model.empty()) {
        settings_.model = model;
    }
}

void GpuBridge::setDevice(const std::string& device) {
    if (!device.empty()) {
        settings_.device = device;
    }
}

bool GpuBridge::score(const std::vector<FeatureVector>& features, std::vector<int>& scores) const {
    if (!settings_.enabled || features.empty()) {
        return false;
    }

    const std::filesystem::path inputPath = tempPath("_features.tsv");
    const std::filesystem::path outputPath = tempPath("_scores.tsv");
    if (!writeFeatureFile(inputPath, features)) {
        return false;
    }

    std::ostringstream arguments;
    arguments << ' ' << quoteArg(settings_.script)
            << " score --input " << quoteArg(inputPath.string())
            << " --output " << quoteArg(outputPath.string())
            << " --model " << quoteArg(settings_.model)
            << " --device " << quoteArg(settings_.device);
    const int exitCode = std::system(buildCommand(settings_.python, arguments.str(), nullRedirect()).c_str());
    const bool ok = exitCode == 0 && readScores(outputPath, scores, features.size());

    std::error_code ignored;
    std::filesystem::remove(inputPath, ignored);
    std::filesystem::remove(outputPath, ignored);
    return ok;
}

bool GpuBridge::train(const std::string& trainingDataPath) const {
    if (!settings_.trainOnGameEnd || trainingDataPath.empty()) {
        return false;
    }

    std::ostringstream arguments;
    arguments << ' ' << quoteArg(settings_.script)
            << " train --data " << quoteArg(trainingDataPath)
            << " --model " << quoteArg(settings_.model)
            << " --device " << quoteArg(settings_.device);
    return std::system(buildCommand(settings_.python, arguments.str(), appendLogRedirect("gpu_train.log")).c_str()) == 0;
}

} // namespace shogi
