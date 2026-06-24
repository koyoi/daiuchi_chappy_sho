#pragma once

#include "evaluation.h"
#include "shogi_types.h"

#include <string>
#include <vector>

namespace shogi {

struct RecordedPly {
    Board before;
    Move move;
    Color mover = Black;
    bool engineMove = false;
};

class OnlineLearner {
public:
    explicit OnlineLearner(Evaluator& evaluator);

    void setEnabled(bool enabled);
    void setRecordOnly(bool recordOnly);
    void setLearningRate(double learningRate);
    void setWeightsPath(const std::string& path);
    void setTrainingDataPath(const std::string& path);
    const std::string& weightsPath() const;
    const std::string& trainingDataPath() const;

    bool loadWeights();
    void saveWeights() const;
    void clearGame();
    void recordMove(const Board& before, const Move& move, bool engineMove);
    void finishGame(int engineResult, Color engineSide);

private:
    FeatureVector averageSuccessorFeatures(const Board& before, Color perspective) const;
    void appendTrainingSample(double label, const FeatureVector& features) const;

    Evaluator& evaluator_;
    std::vector<RecordedPly> game_;
    std::string weightsPath_ = "linear.weights";
    std::string trainingDataPath_ = "mlp_training.tsv";
    double learningRate_ = 1.5;
    bool enabled_ = true;
    bool recordOnly_ = false;
};

} // namespace shogi
