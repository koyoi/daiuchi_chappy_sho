#include "kifu_learner.h"

#include "evaluation.h"
#include "notation.h"
#include "position.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace shogi {

namespace {

struct TrainingSample {
    std::string sfen;
    std::string usiMove;
};

std::vector<TrainingSample> loadTrainingFile(const std::string& path) {
    std::vector<TrainingSample> samples;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open training file: " << path << std::endl;
        return samples;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        samples.push_back({line.substr(0, tab), line.substr(tab + 1)});
    }
    std::cerr << "Loaded " << samples.size() << " training samples" << std::endl;
    return samples;
}

} // namespace

int learnFromKifu(const KifuLearnConfig& config) {
    Evaluator evaluator;
    if (evaluator.load(config.weightsPath)) {
        std::cerr << "Loaded weights from " << config.weightsPath << std::endl;
    } else {
        std::cerr << "Starting with default weights" << std::endl;
    }
    evaluator.setHeavyFeatures(false);

    auto samples = loadTrainingFile(config.trainingFile);
    if (samples.empty()) return 1;

    std::mt19937 rng{std::random_device{}()};
    int totalUpdates = 0;
    int totalSkipped = 0;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(samples.begin(), samples.end(), rng);
        auto epochStart = std::chrono::steady_clock::now();
        int epochUpdates = 0;
        int epochSkipped = 0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const auto& sample = samples[i];

            Board board;
            std::istringstream iss(sample.sfen);
            std::string boardPart, sidePart, handPart, movePart;
            iss >> boardPart >> sidePart >> handPart >> movePart;
            if (!setFromSfen(board, boardPart, sidePart, handPart, movePart)) {
                ++epochSkipped;
                continue;
            }

            Move move = parseUsiMove(board, sample.usiMove);
            if (move.to < 0) {
                ++epochSkipped;
                continue;
            }

            if (evaluator.learnFromMove(board, move, config.learningRate)) {
                ++epochUpdates;
            } else {
                ++epochSkipped;
            }

            if (config.saveInterval > 0 && (epochUpdates % config.saveInterval) == 0 && epochUpdates > 0) {
                evaluator.save(config.weightsPath);
            }

            if ((i + 1) % 10000 == 0) {
                int pct = static_cast<int>((i + 1) * 100 / samples.size());
                std::cerr << "\r  epoch " << (epoch + 1) << "/" << config.epochs
                          << ": " << pct << "% (" << epochUpdates << " updates, "
                          << epochSkipped << " skipped)" << std::flush;
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - epochStart;
        double secs = std::chrono::duration<double>(elapsed).count();
        totalUpdates += epochUpdates;
        totalSkipped += epochSkipped;

        std::cerr << "\r  epoch " << (epoch + 1) << "/" << config.epochs
                  << ": " << epochUpdates << " updates, " << epochSkipped
                  << " skipped (" << secs << "s)" << std::endl;

        evaluator.save(config.weightsPath);
    }

    std::cerr << "Training complete: " << totalUpdates << " updates, "
              << totalSkipped << " skipped over " << config.epochs << " epochs"
              << std::endl;
    return 0;
}

} // namespace shogi
