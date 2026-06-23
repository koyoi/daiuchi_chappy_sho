#include "onnx_inference.h"
#include "movegen.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace shogi {

struct OnnxInference::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "shogi"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    bool loaded = false;
};

OnnxInference::OnnxInference() : impl_(std::make_unique<Impl>()) {}

OnnxInference::~OnnxInference() = default;

bool OnnxInference::loadModel(const std::string& modelPath, const std::string& device) {
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device != "cpu") {
            try {
                OrtCUDAProviderOptions cudaOpts{};
                cudaOpts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cudaOpts);
            } catch (...) {
                // CUDA not available, fall back to CPU
            }
        }

        impl_->session = std::make_unique<Ort::Session>(impl_->env, modelPath.c_str(), opts);
        impl_->loaded = true;
        lastError_.clear();
        return true;
    } catch (const Ort::Exception& e) {
        lastError_ = e.what();
        impl_->loaded = false;
        return false;
    }
}

bool OnnxInference::isLoaded() const {
    return impl_ && impl_->loaded;
}

void OnnxInference::encodeBoardDirect(const Board& board, int batchIdx,
                                       std::vector<int64_t>& boardData,
                                       std::vector<int64_t>& handData,
                                       std::vector<int64_t>& sideData,
                                       std::vector<int64_t>& ownAtkData,
                                       std::vector<int64_t>& oppAtkData) const {
    const int bOff = batchIdx * 81;
    const int hOff = batchIdx * 14;

    for (int i = 0; i < 81; ++i) {
        int piece = board.squares[i];
        if (piece == 0) {
            boardData[bOff + i] = 0;
        } else if (piece > 0) {
            boardData[bOff + i] = std::min(piece, 28);
        } else {
            boardData[bOff + i] = std::min(14 + (-piece), 28);
        }
    }

    for (int pt = 1; pt <= 7; ++pt) {
        handData[hOff + pt - 1] = std::min(board.blackHand[pt], 19);
    }
    for (int pt = 1; pt <= 7; ++pt) {
        handData[hOff + 7 + pt - 1] = std::min(board.whiteHand[pt], 19);
    }

    sideData[batchIdx] = (board.side == Black) ? 0 : 1;

    Color own = board.side;
    Color opp = (own == Black) ? White : Black;
    for (int sq = 0; sq < 81; ++sq) {
        ownAtkData[bOff + sq] = std::min(countAttackers(board, sq, own), 8);
        oppAtkData[bOff + sq] = std::min(countAttackers(board, sq, opp), 8);
    }
}

NNOutput OnnxInference::evaluate(const Board& board) {
    auto results = evaluateBatch({board});
    return results.empty() ? NNOutput{} : results[0];
}

std::vector<NNOutput> OnnxInference::evaluateBatch(const std::vector<Board>& boards) {
    if (!impl_->loaded || boards.empty()) {
        std::vector<NNOutput> fallback(boards.size());
        for (auto& o : fallback) {
            o.value = 0.0;
            o.policy.assign(PolicySize, 1.0 / PolicySize);
        }
        return fallback;
    }

    const int B = static_cast<int>(boards.size());

    std::vector<int64_t> boardData(B * 81, 0);
    std::vector<int64_t> handData(B * 14, 0);
    std::vector<int64_t> sideData(B, 0);
    std::vector<int64_t> ownAtkData(B * 81, 0);
    std::vector<int64_t> oppAtkData(B * 81, 0);

    for (int i = 0; i < B; ++i) {
        encodeBoardDirect(boards[i], i, boardData, handData, sideData, ownAtkData, oppAtkData);
    }

    std::array<int64_t, 2> boardShape = {B, 81};
    std::array<int64_t, 2> handShape = {B, 14};
    std::array<int64_t, 1> sideShape = {B};
    std::array<int64_t, 2> atkShape = {B, 81};

    std::array<Ort::Value, 5> inputTensors = {
        Ort::Value::CreateTensor<int64_t>(impl_->memoryInfo, boardData.data(), boardData.size(),
                                          boardShape.data(), boardShape.size()),
        Ort::Value::CreateTensor<int64_t>(impl_->memoryInfo, handData.data(), handData.size(),
                                          handShape.data(), handShape.size()),
        Ort::Value::CreateTensor<int64_t>(impl_->memoryInfo, sideData.data(), sideData.size(),
                                          sideShape.data(), sideShape.size()),
        Ort::Value::CreateTensor<int64_t>(impl_->memoryInfo, ownAtkData.data(), ownAtkData.size(),
                                          atkShape.data(), atkShape.size()),
        Ort::Value::CreateTensor<int64_t>(impl_->memoryInfo, oppAtkData.data(), oppAtkData.size(),
                                          atkShape.data(), atkShape.size()),
    };

    const char* inputNames[] = {"board_tokens", "hand_tokens", "side_token", "own_atk", "opp_atk"};
    const char* outputNames[] = {"value", "policy_logits"};

    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                       inputNames, inputTensors.data(), 5,
                                       outputNames, 2);

    const float* valueData = outputs[0].GetTensorData<float>();
    const float* policyData = outputs[1].GetTensorData<float>();

    std::vector<NNOutput> results(B);
    for (int i = 0; i < B; ++i) {
        results[i].value = static_cast<double>(valueData[i]);

        // Softmax over policy logits
        results[i].policy.resize(PolicySize);
        const float* logits = policyData + i * PolicySize;
        double maxLogit = *std::max_element(logits, logits + PolicySize);
        double expSum = 0.0;
        for (int j = 0; j < PolicySize; ++j) {
            results[i].policy[j] = std::exp(static_cast<double>(logits[j]) - maxLogit);
            expSum += results[i].policy[j];
        }
        for (int j = 0; j < PolicySize; ++j) {
            results[i].policy[j] /= expSum;
        }
    }

    return results;
}

} // namespace shogi
