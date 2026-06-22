#pragma once

#include <string>

namespace shogi {

struct KifuLearnConfig {
    std::string trainingFile;
    std::string weightsPath = "random-shogi.weights";
    double learningRate = 0.01;
    int epochs = 1;
    int batchSize = 256;
    double temperature = 100.0;
};

int learnFromKifu(const KifuLearnConfig& config);

} // namespace shogi
