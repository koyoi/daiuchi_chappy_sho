#pragma once

#include "shogi_types.h"

#include <array>
#include <string>

namespace shogi {

constexpr int FeatureCount = 74;
using FeatureVector = std::array<double, FeatureCount>;

class Evaluator {
public:
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

private:
    std::array<double, FeatureCount> weights_{};
    bool heavyFeatures_ = false;
};

FeatureVector operator-(const FeatureVector& left, const FeatureVector& right);
FeatureVector& operator+=(FeatureVector& left, const FeatureVector& right);
FeatureVector& operator/=(FeatureVector& left, double value);

} // namespace shogi
