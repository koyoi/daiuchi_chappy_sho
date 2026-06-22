#include "learning.h"

#include "movegen.h"
#include "position.h"

#include <fstream>

namespace shogi {

OnlineLearner::OnlineLearner(Evaluator& evaluator)
    : evaluator_(evaluator) {
}

void OnlineLearner::setEnabled(bool enabled) {
    enabled_ = enabled;
}

void OnlineLearner::setRecordOnly(bool recordOnly) {
    recordOnly_ = recordOnly;
}

void OnlineLearner::setLearningRate(double learningRate) {
    learningRate_ = learningRate;
}

void OnlineLearner::setWeightsPath(const std::string& path) {
    if (!path.empty()) {
        weightsPath_ = path;
    }
}

void OnlineLearner::setTrainingDataPath(const std::string& path) {
    if (!path.empty()) {
        trainingDataPath_ = path;
    }
}

const std::string& OnlineLearner::weightsPath() const {
    return weightsPath_;
}

const std::string& OnlineLearner::trainingDataPath() const {
    return trainingDataPath_;
}

bool OnlineLearner::loadWeights() {
    return evaluator_.load(weightsPath_);
}

void OnlineLearner::saveWeights() const {
    evaluator_.save(weightsPath_);
}

void OnlineLearner::clearGame() {
    game_.clear();
}

void OnlineLearner::recordMove(const Board& before, const Move& move, bool engineMove) {
    if (!enabled_ || move.to < 0) {
        return;
    }
    game_.push_back(RecordedPly{before, move, before.side, engineMove});
}

FeatureVector OnlineLearner::averageSuccessorFeatures(const Board& before, Color perspective) const {
    FeatureVector average{};
    const auto legal = generateLegalMoves(before, true);
    if (legal.empty()) {
        return average;
    }
    for (const Move& candidate : legal) {
        Board next = before;
        applyMove(next, candidate);
        average += evaluator_.extractFeatures(next, perspective);
    }
    average /= static_cast<double>(legal.size());
    return average;
}

void OnlineLearner::appendTrainingSample(double label, const FeatureVector& features) const {
    std::ofstream output(trainingDataPath_, std::ios::app);
    if (!output) {
        return;
    }
    output << label;
    for (double value : features) {
        output << '\t' << value;
    }
    output << '\n';
}

void OnlineLearner::finishGame(int engineResult, Color engineSide) {
    if (!enabled_ || engineResult == 0 || game_.empty()) {
        clearGame();
        return;
    }

    const Color winner = engineResult > 0 ? engineSide : opposite(engineSide);
    for (std::size_t i = 0; i < game_.size(); ++i) {
        const RecordedPly& ply = game_[i];
        Board after = ply.before;
        applyMove(after, ply.move);

        const FeatureVector chosen = evaluator_.extractFeatures(after, ply.mover);
        const FeatureVector baseline = averageSuccessorFeatures(ply.before, ply.mover);
        const FeatureVector delta = chosen - baseline;

        const double outcomeForMover = ply.mover == winner ? 1.0 : -1.0;
        const double progress = game_.size() <= 1 ? 1.0 : static_cast<double>(i) / static_cast<double>(game_.size() - 1);
        if (!recordOnly_) {
            const double actorScale = ply.engineMove ? 1.0 : 0.65;
            const double recencyScale = 0.35 + 1.65 * progress;
            evaluator_.applyDelta(delta, learningRate_ * actorScale * recencyScale * outcomeForMover);
        }
        const double confidence = 0.55 + 0.40 * progress;
        const double smoothLabel = outcomeForMover > 0 ? confidence : (1.0 - confidence);
        appendTrainingSample(smoothLabel, chosen);
    }

    if (!recordOnly_) {
        saveWeights();
    }
    clearGame();
}

} // namespace shogi
