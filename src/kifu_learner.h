#pragma once

#include <string>

namespace shogi {

struct KifuLearnConfig {
    std::string trainingFile;
    std::string weightsPath = "random-shogi.weights";
    double learningRate = 0.01;
    int epochs = 1;
    int saveInterval = 10000;
};

int learnFromKifu(const KifuLearnConfig& config);

} // namespace shogi
