#include "nn_bridge.h"
#include "movegen.h"
#include "onnx_inference.h"

#include <algorithm>
#include <sstream>

namespace shogi {

// --- Move index encoding ---
// 10 directions: UP,DOWN,LEFT,RIGHT,UL,UR,DL,DR,KnightL,KnightR
// Channels 0-9: move without promotion (direction encodes from->to delta)
// Channels 10-19: move with promotion
// Channels 20-26: drop (Pawn,Lance,Knight,Silver,Gold,Bishop,Rook)
//
// Index = toSquare * 27 + channel

namespace {

constexpr int DirUp = 0;
constexpr int DirDown = 1;
constexpr int DirLeft = 2;
constexpr int DirRight = 3;
constexpr int DirUpLeft = 4;
constexpr int DirUpRight = 5;
constexpr int DirDownLeft = 6;
constexpr int DirDownRight = 7;
constexpr int DirKnightLeft = 8;
constexpr int DirKnightRight = 9;

int directionOf(int fromSq, int toSq, Color side) {
    int ff = fileOf(fromSq), fr = rankOf(fromSq);
    int tf = fileOf(toSq), tr = rankOf(toSq);
    int df = tf - ff;
    int dr = tr - fr;
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

} // namespace

int NNBridge::moveToIndex(const Move& move, Color side) {
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

NNBridge::NNBridge() = default;

NNBridge::~NNBridge() {
    shutdown();
}

void NNBridge::setEnabled(bool enabled) { settings_.enabled = enabled; }
bool NNBridge::enabled() const { return settings_.enabled; }
void NNBridge::setModel(const std::string& m) {
    if (!m.empty() && m != settings_.model) {
        shutdown();
        settings_.model = m;
        onnx_.reset();
        onnxAttempted_ = false;
    }
}
void NNBridge::setDevice(const std::string& d) { if (!d.empty()) settings_.device = d; }

bool NNBridge::isReady() const {
    return onnx_ && onnx_->isLoaded();
}

std::string NNBridge::deviceUsed() const {
    if (onnx_ && onnx_->isLoaded()) return onnx_->deviceUsed();
    return "none";
}

std::string NNBridge::cudaError() const {
    if (onnx_) return onnx_->cudaError();
    return {};
}

NNOutput NNBridge::makeFallbackOutput() const {
    NNOutput out;
    out.value = 0.0;
    out.policy.assign(PolicySize, 1.0 / PolicySize);
    return out;
}

bool NNBridge::ensureReady() {
    if (onnxAttempted_) return onnx_ && onnx_->isLoaded();
    onnxAttempted_ = true;

    std::string onnxPath = settings_.model;
    auto dotPos = onnxPath.rfind('.');
    if (dotPos != std::string::npos) {
        std::string ext = onnxPath.substr(dotPos);
        if (ext != ".onnx") {
            onnxPath = onnxPath.substr(0, dotPos) + ".onnx";
        }
    } else {
        onnxPath += ".onnx";
    }

    onnx_ = std::make_unique<OnnxInference>();
    if (onnx_->loadModel(onnxPath, settings_.device)) {
        lastError_.clear();
        return true;
    }
    lastError_ = "ONNX load failed: " + onnx_->lastError();
    onnx_.reset();
    return false;
}

void NNBridge::shutdown() {
    onnx_.reset();
    onnxAttempted_ = false;
}

NNOutput NNBridge::evaluate(const Board& board) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureReady()) return makeFallbackOutput();
    return onnx_->evaluate(board);
}

std::vector<NNOutput> NNBridge::evaluateBatch(const std::vector<Board>& boards) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureReady() || boards.empty()) {
        std::vector<NNOutput> fallback(boards.size());
        for (auto& o : fallback) o = makeFallbackOutput();
        return fallback;
    }
    return onnx_->evaluateBatch(boards);
}

} // namespace shogi
