#include "alpha_onnx_inference.h"
#include "movegen.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <thread>

namespace shogi {

namespace {

constexpr int DirUp = 0, DirDown = 1, DirLeft = 2, DirRight = 3;
constexpr int DirUpLeft = 4, DirUpRight = 5, DirDownLeft = 6, DirDownRight = 7;
constexpr int DirKnightLeft = 8, DirKnightRight = 9;

int fileOf(int sq) { return sq % 9; }
int rankOf(int sq) { return sq / 9; }

int directionOf(int fromSq, int toSq, Color side) {
    int df = fileOf(toSq) - fileOf(fromSq);
    int dr = rankOf(toSq) - rankOf(fromSq);
    if (side == White) { df = -df; dr = -dr; }
    if (df == 0 && dr < 0) return DirUp;
    if (df == 0 && dr > 0) return DirDown;
    if (dr == 0 && df < 0) return DirLeft;
    if (dr == 0 && df > 0) return DirRight;
    if (df < 0 && dr < 0) return DirUpLeft;
    if (df > 0 && dr < 0) return DirUpRight;
    if (df < 0 && dr > 0) return DirDownLeft;
    if (df > 0 && dr > 0) return DirDownRight;
    if (df == -1 && dr == -2) return DirKnightLeft;
    if (df == 1 && dr == -2) return DirKnightRight;
    return -1;
}

constexpr int HandMaxCounts[7] = {18, 4, 4, 4, 4, 2, 2};

} // namespace

struct AlphaOnnxInference::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "alpha"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    bool loaded = false;
};

AlphaOnnxInference::AlphaOnnxInference() : impl_(std::make_unique<Impl>()) {}
AlphaOnnxInference::~AlphaOnnxInference() = default;

bool AlphaOnnxInference::loadModel(const std::string& modelPath, const std::string& device) {
    modelPath_ = modelPath;
    try {
        Ort::SessionOptions opts;
        if (device == "cpu") {
            opts.SetIntraOpNumThreads(
                std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
        } else {
            opts.SetIntraOpNumThreads(1);
        }
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device != "cpu") {
            try {
                OrtCUDAProviderOptions cudaOpts{};
                cudaOpts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cudaOpts);
            } catch (...) {}
        }

        std::wstring wpath(modelPath.begin(), modelPath.end());
        impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(), opts);
        impl_->loaded = true;
        lastError_.clear();
        return true;
    } catch (const Ort::Exception& e) {
        lastError_ = e.what();
        impl_->loaded = false;
        return false;
    }
}

bool AlphaOnnxInference::isLoaded() const {
    return impl_ && impl_->loaded;
}

int AlphaOnnxInference::moveToIndex(const Move& move, Color side) {
    if (move.isDrop()) {
        int dropChannel = 20 + (static_cast<int>(move.drop) - 1);
        if (dropChannel < 20 || dropChannel > 26) return -1;
        return move.to * 27 + dropChannel;
    }
    int dir = directionOf(move.from, move.to, side);
    if (dir < 0) return -1;
    int channel = move.promote ? (dir + 10) : dir;
    return move.to * 27 + channel;
}

void AlphaOnnxInference::encodeBoardSpatial(const Board& board, int batchIdx,
                                             std::vector<float>& features) const {
    const int planeSize = 81;
    const int batchOffset = batchIdx * AlphaInputChannels * planeSize;

    // ch 0-27: piece planes (14 types x 2 colors)
    for (int sq = 0; sq < 81; ++sq) {
        int piece = board.squares[sq];
        if (piece == 0) continue;
        int ch;
        if (piece > 0) {
            ch = piece - 1;          // Black pieces: ch 0-13
        } else {
            ch = 14 + (-piece) - 1;  // White pieces: ch 14-27
        }
        features[batchOffset + ch * planeSize + sq] = 1.0f;
    }

    // ch 28-34: Black hand (7 types, normalized)
    for (int i = 0; i < 7; ++i) {
        float val = static_cast<float>(board.blackHand[i + 1]) / HandMaxCounts[i];
        int ch = 28 + i;
        for (int sq = 0; sq < planeSize; ++sq)
            features[batchOffset + ch * planeSize + sq] = val;
    }

    // ch 35-41: White hand (7 types, normalized)
    for (int i = 0; i < 7; ++i) {
        float val = static_cast<float>(board.whiteHand[i + 1]) / HandMaxCounts[i];
        int ch = 35 + i;
        for (int sq = 0; sq < planeSize; ++sq)
            features[batchOffset + ch * planeSize + sq] = val;
    }

    // ch 42-43: attack count planes
    Color own = board.side;
    Color opp = (own == Black) ? White : Black;
    for (int sq = 0; sq < 81; ++sq) {
        features[batchOffset + 42 * planeSize + sq] =
            std::min(countAttackers(board, sq, own), 8) / 8.0f;
        features[batchOffset + 43 * planeSize + sq] =
            std::min(countAttackers(board, sq, opp), 8) / 8.0f;
    }

    // ch 44: side to move (1.0 = Black)
    float sideVal = (board.side == Black) ? 1.0f : 0.0f;
    for (int sq = 0; sq < planeSize; ++sq)
        features[batchOffset + 44 * planeSize + sq] = sideVal;
}

AlphaNNOutput AlphaOnnxInference::evaluate(const Board& board) {
    auto results = evaluateBatch({board});
    return results.empty() ? AlphaNNOutput{} : results[0];
}

std::vector<AlphaNNOutput> AlphaOnnxInference::evaluateBatch(
        const std::vector<Board>& boards) {
    if (!impl_->loaded || boards.empty()) {
        std::vector<AlphaNNOutput> fallback(boards.size());
        for (auto& o : fallback) {
            o.value = 0.0;
            o.wdl = {0.33, 0.34, 0.33};
            o.policy.assign(AlphaPolicySize, 1.0 / AlphaPolicySize);
        }
        return fallback;
    }

    const int B = static_cast<int>(boards.size());
    const int totalFloats = B * AlphaInputChannels * 81;
    std::vector<float> featData(totalFloats, 0.0f);

    for (int i = 0; i < B; ++i) {
        encodeBoardSpatial(boards[i], i, featData);
    }

    std::array<int64_t, 4> featShape = {B, AlphaInputChannels, 9, 9};
    Ort::Value featTensor = Ort::Value::CreateTensor<float>(
        impl_->memoryInfo, featData.data(), featData.size(),
        featShape.data(), featShape.size());

    const char* inputNames[] = {"features"};
    const char* outputNames[] = {"value_wdl", "policy_logits"};

    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        inputNames, &featTensor, 1,
        outputNames, 2);

    const float* wdlData = outputs[0].GetTensorData<float>();
    const float* policyData = outputs[1].GetTensorData<float>();

    std::vector<AlphaNNOutput> results(B);
    for (int i = 0; i < B; ++i) {
        // WDL softmax
        const float* wdl = wdlData + i * 3;
        float wdlMax = std::max({wdl[0], wdl[1], wdl[2]});
        double wdlExp[3];
        double wdlSum = 0.0;
        for (int j = 0; j < 3; ++j) {
            wdlExp[j] = std::exp(static_cast<double>(wdl[j]) - wdlMax);
            wdlSum += wdlExp[j];
        }
        for (int j = 0; j < 3; ++j) {
            results[i].wdl[j] = wdlExp[j] / wdlSum;
        }
        results[i].value = results[i].wdl[0] - results[i].wdl[2];

        // Policy softmax
        results[i].policy.resize(AlphaPolicySize);
        const float* logits = policyData + i * AlphaPolicySize;
        double maxLogit = *std::max_element(logits, logits + AlphaPolicySize);
        double expSum = 0.0;
        for (int j = 0; j < AlphaPolicySize; ++j) {
            results[i].policy[j] = std::exp(static_cast<double>(logits[j]) - maxLogit);
            expSum += results[i].policy[j];
        }
        for (int j = 0; j < AlphaPolicySize; ++j) {
            results[i].policy[j] /= expSum;
        }
    }

    return results;
}

} // namespace shogi
