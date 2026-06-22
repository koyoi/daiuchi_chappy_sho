#include "csa_protocol.h"
#include "kifu_learner.h"
#include "usi_protocol.h"

#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        if (executable.has_parent_path()) {
            std::filesystem::current_path(executable.parent_path());
        }
    } catch (...) {
    }

    std::string protocol = "usi";
    shogi::KifuLearnConfig learnConfig;
    shogi::ExtractFeaturesConfig extractConfig;
    bool learnMode = false;
    bool extractMode = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--protocol" || arg == "-p") && i + 1 < argc) {
            protocol = argv[++i];
        } else if (arg == "--csa") {
            protocol = "csa";
        } else if (arg == "--usi") {
            protocol = "usi";
        } else if (arg == "--learn" && i + 1 < argc) {
            learnMode = true;
            learnConfig.trainingFile = argv[++i];
        } else if (arg == "--extract-features" && i + 1 < argc) {
            extractMode = true;
            extractConfig.trainingFile = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            extractConfig.outputFile = argv[++i];
        } else if (arg == "--negatives" && i + 1 < argc) {
            extractConfig.negatives = std::stoi(argv[++i]);
        } else if (arg == "--weights" && i + 1 < argc) {
            learnConfig.weightsPath = argv[++i];
        } else if (arg == "--lr" && i + 1 < argc) {
            learnConfig.learningRate = std::stod(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            learnConfig.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            learnConfig.batchSize = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            learnConfig.temperature = std::stod(argv[++i]);
        }
    }

    if (extractMode) {
        return shogi::extractFeatures(extractConfig);
    } else if (learnMode) {
        return shogi::learnFromKifu(learnConfig);
    } else if (protocol == "csa") {
        shogi::csaLoop();
    } else {
        shogi::usiLoop();
    }
    return 0;
}
