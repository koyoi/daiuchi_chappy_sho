#include "kifu_learner.h"

#include "evaluation.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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

struct BatchResult {
    FeatureVector delta{};
    double totalLoss = 0.0;
    int correct = 0;
    int updates = 0;
    int skipped = 0;
};

void processBatchSlice(const Evaluator& evaluator,
                       const std::vector<TrainingSample>& samples,
                       std::size_t begin, std::size_t end,
                       double temperature,
                       BatchResult& result) {
    for (std::size_t i = begin; i < end; ++i) {
        const auto& sample = samples[i];

        Board board;
        std::istringstream iss(sample.sfen);
        std::string boardPart, sidePart, handPart, movePart;
        iss >> boardPart >> sidePart >> handPart >> movePart;
        if (!setFromSfen(board, boardPart, sidePart, handPart, movePart)) {
            ++result.skipped;
            continue;
        }

        Move move = parseUsiMove(board, sample.usiMove);
        if (move.to < 0) {
            ++result.skipped;
            continue;
        }

        Evaluator::GradientResult grad;
        if (evaluator.computeGradient(board, move, grad, temperature)) {
            result.delta += grad.delta;
            result.totalLoss += grad.loss;
            if (grad.correct) ++result.correct;
            ++result.updates;
        } else {
            ++result.skipped;
        }
    }
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

    const int numThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    const std::size_t batchSize = static_cast<std::size_t>(config.batchSize);
    std::cerr << "Threads: " << numThreads << ", batch size: " << batchSize
              << ", temperature: " << config.temperature << std::endl;

    std::mt19937 rng{std::random_device{}()};
    int totalUpdates = 0;
    int totalSkipped = 0;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(samples.begin(), samples.end(), rng);
        auto epochStart = std::chrono::steady_clock::now();
        int epochUpdates = 0;
        int epochSkipped = 0;
        int epochCorrect = 0;
        double epochLoss = 0.0;

        for (std::size_t batchStart = 0; batchStart < samples.size(); batchStart += batchSize) {
            std::size_t batchEnd = std::min(batchStart + batchSize, samples.size());
            std::size_t batchLen = batchEnd - batchStart;

            std::vector<BatchResult> results(numThreads);
            std::vector<std::thread> threads;

            std::size_t perThread = (batchLen + numThreads - 1) / numThreads;
            for (int t = 0; t < numThreads; ++t) {
                std::size_t tBegin = batchStart + t * perThread;
                std::size_t tEnd = std::min(tBegin + perThread, batchEnd);
                if (tBegin >= batchEnd) break;
                threads.emplace_back(processBatchSlice,
                    std::cref(evaluator), std::cref(samples),
                    tBegin, tEnd, config.temperature, std::ref(results[t]));
            }
            for (auto& th : threads) th.join();

            FeatureVector batchDelta{};
            int batchUpdates = 0;
            for (auto& r : results) {
                batchDelta += r.delta;
                batchUpdates += r.updates;
                epochSkipped += r.skipped;
                epochCorrect += r.correct;
                epochLoss += r.totalLoss;
            }
            if (batchUpdates > 0) {
                batchDelta /= static_cast<double>(batchUpdates);
                evaluator.applyDelta(batchDelta, config.learningRate);
                epochUpdates += batchUpdates;
            }

            std::size_t processed = batchEnd;
            if (processed % 50000 < batchSize || processed == samples.size()) {
                double avgLoss = epochUpdates > 0 ? epochLoss / epochUpdates : 0.0;
                double acc = epochUpdates > 0 ? 100.0 * epochCorrect / epochUpdates : 0.0;
                int pct = static_cast<int>(processed * 100 / samples.size());
                std::cerr << "\r  epoch " << (epoch + 1) << "/" << config.epochs
                          << ": " << pct << "% | loss=" << std::fixed << std::setprecision(4) << avgLoss
                          << " acc=" << std::fixed << std::setprecision(1) << acc << "%"
                          << std::defaultfloat << std::flush;
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - epochStart;
        double secs = std::chrono::duration<double>(elapsed).count();
        double avgLoss = epochUpdates > 0 ? epochLoss / epochUpdates : 0.0;
        double acc = epochUpdates > 0 ? 100.0 * epochCorrect / epochUpdates : 0.0;
        totalUpdates += epochUpdates;
        totalSkipped += epochSkipped;

        std::cerr << "\r  epoch " << (epoch + 1) << "/" << config.epochs
                  << ": loss=" << std::fixed << std::setprecision(4) << avgLoss
                  << " acc=" << std::fixed << std::setprecision(1) << acc << "% "
                  << std::defaultfloat
                  << epochUpdates << " updates (" << secs << "s)" << std::endl;

        evaluator.save(config.weightsPath);
    }

    std::cerr << "Training complete: " << totalUpdates << " updates, "
              << totalSkipped << " skipped over " << config.epochs << " epochs"
              << std::endl;
    return 0;
}

int extractFeatures(const ExtractFeaturesConfig& config) {
    Evaluator evaluator;
    if (evaluator.load("linear.weights")) {
        std::cerr << "Loaded weights for eval bootstrap" << std::endl;
    }
    evaluator.setHeavyFeatures(false);

    auto samples = loadTrainingFile(config.trainingFile);
    if (samples.empty()) return 1;

    std::ofstream out(config.outputFile);
    if (!out) {
        std::cerr << "Cannot open output file: " << config.outputFile << std::endl;
        return 1;
    }

    constexpr double evalLambda = 0.6;
    constexpr double evalScale = 361.0;

    std::mt19937 rng{std::random_device{}()};
    int written = 0;
    int skipped = 0;

    for (const auto& sample : samples) {
        Board board;
        std::istringstream iss(sample.sfen);
        std::string boardPart, sidePart, handPart, movePart;
        iss >> boardPart >> sidePart >> handPart >> movePart;
        if (!setFromSfen(board, boardPart, sidePart, handPart, movePart)) {
            ++skipped;
            continue;
        }

        Move move = parseUsiMove(board, sample.usiMove);
        if (move.to < 0) {
            ++skipped;
            continue;
        }

        Color perspective = board.side;

        Board after = board;
        applyMove(after, move);
        FeatureVector posFeatures = evaluator.extractFeatures(after, perspective);

        double eval = static_cast<double>(evaluator.evaluate(after, perspective));
        double evalWp = 1.0 / (1.0 + std::exp(-eval / evalScale));
        double posLabel = std::clamp(evalLambda * evalWp + (1.0 - evalLambda) * 0.85, 0.01, 0.99);
        out << posLabel;
        for (int i = 0; i < FeatureCount; ++i) {
            out << '\t' << posFeatures[i];
        }
        out << '\n';
        ++written;

        auto legal = generateLegalMoves(board, true);
        std::vector<Move> others;
        for (const Move& m : legal) {
            if (!sameMove(m, move)) others.push_back(m);
        }
        if (!others.empty()) {
            std::shuffle(others.begin(), others.end(), rng);
            int negCount = std::min(config.negatives, static_cast<int>(others.size()));
            for (int n = 0; n < negCount; ++n) {
                Board neg = board;
                applyMove(neg, others[n]);
                FeatureVector negFeatures = evaluator.extractFeatures(neg, perspective);
                double negEval = static_cast<double>(evaluator.evaluate(neg, perspective));
                double negWp = 1.0 / (1.0 + std::exp(-negEval / evalScale));
                double negLabel = std::clamp(evalLambda * negWp + (1.0 - evalLambda) * 0.15, 0.01, 0.99);
                out << negLabel;
                for (int i = 0; i < FeatureCount; ++i) {
                    out << '\t' << negFeatures[i];
                }
                out << '\n';
                ++written;
            }
        }

        if (written % 100000 < 3) {
            std::cerr << "\r  extracted " << written << " samples..." << std::flush;
        }
    }

    std::cerr << "\r  extracted " << written << " samples (skipped " << skipped << ")" << std::endl;
    return 0;
}

} // namespace shogi
