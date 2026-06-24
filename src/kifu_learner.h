#pragma once

#include <string>

namespace shogi {

struct KifuLearnConfig {
    std::string trainingFile;
    std::string weightsPath = "linear.weights";
    double learningRate = 0.01;
    int epochs = 1;
    int batchSize = 256;
    double temperature = 100.0;
};

int learnFromKifu(const KifuLearnConfig& config);

struct ExtractFeaturesConfig {
    std::string trainingFile;
    std::string outputFile = "mlp_training.tsv";
    int negatives = 1;
};

int extractFeatures(const ExtractFeaturesConfig& config);

} // namespace shogi
