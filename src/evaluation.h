#pragma once

#include "shogi_types.h"

#include <array>
#include <string>

namespace shogi {

constexpr int FeatureCount = 100;
using FeatureVector = std::array<double, FeatureCount>;

class Evaluator {
public:
    static constexpr int MlpHidden1 = 128;
    static constexpr int MlpHidden2 = 64;

    Evaluator();

    int evaluate(const Board& board, Color perspective) const;
    FeatureVector extractFeatures(const Board& board, Color perspective) const;
    void setHeavyFeatures(bool enabled);

    bool load(const std::string& path);
    bool save(const std::string& path) const;
    void applyDelta(const FeatureVector& delta, double scale);
    bool learnFromMove(const Board& board, const Move& correctMove, double lr);
    struct GradientResult {
        FeatureVector delta{};
        double loss = 0.0;
        bool correct = false;
    };
    bool computeGradient(const Board& board, const Move& correctMove, GradientResult& out, double temperature = 100.0) const;

    bool loadMlp(const std::string& path);
    bool mlpLoaded() const { return mlpLoaded_; }
    void setUseMlp(bool use) { useMlp_ = use; }
    bool useMlp() const { return useMlp_; }

private:
    int evaluateMlp(const FeatureVector& features) const;

    std::array<double, FeatureCount> weights_{};
    bool heavyFeatures_ = false;

    double w1_[MlpHidden1][FeatureCount]{};
    double b1_[MlpHidden1]{};
    double w2_[MlpHidden2][MlpHidden1]{};
    double b2_[MlpHidden2]{};
    double w3_[MlpHidden2]{};
    double b3_ = 0.0;
    bool mlpLoaded_ = false;
    bool useMlp_ = true;
};

FeatureVector operator-(const FeatureVector& left, const FeatureVector& right);
FeatureVector& operator+=(FeatureVector& left, const FeatureVector& right);
FeatureVector& operator/=(FeatureVector& left, double value);

} // namespace shogi
