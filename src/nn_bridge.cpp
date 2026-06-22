#include "nn_bridge.h"
#include "movegen.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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

std::string NNBridge::encodeBoardState(const Board& board) {
    // Format: 81 square values, then 7 black hand counts, then 7 white hand counts, then side
    // Total: 81 + 7 + 7 + 1 = 96 integers, space-separated
    std::ostringstream oss;
    for (int i = 0; i < 81; ++i) {
        if (i > 0) oss << ' ';
        int piece = board.squares[i];
        // Encode: 0=empty, 1-14=black pieces, 15-28=white pieces
        if (piece == 0) {
            oss << 0;
        } else if (piece > 0) {
            oss << piece; // Black: 1-14
        } else {
            oss << (14 + (-piece)); // White: 15-28
        }
    }
    // Hand pieces (Pawn through Rook, indices 1-7)
    for (int pt = 1; pt <= 7; ++pt) {
        oss << ' ' << board.blackHand[pt];
    }
    for (int pt = 1; pt <= 7; ++pt) {
        oss << ' ' << board.whiteHand[pt];
    }
    oss << ' ' << (board.side == Black ? 0 : 1);
    // Own and opponent attack counts per square (81 + 81 = 162 values)
    Color own = board.side;
    Color opp = (own == Black) ? White : Black;
    for (int sq = 0; sq < 81; ++sq) {
        oss << ' ' << std::min(8, countAttackers(board, sq, own));
    }
    for (int sq = 0; sq < 81; ++sq) {
        oss << ' ' << std::min(8, countAttackers(board, sq, opp));
    }
    return oss.str();
}

// --- Process management (platform-specific) ---

NNBridge::NNBridge() = default;

NNBridge::~NNBridge() {
    shutdown();
}

void NNBridge::setEnabled(bool enabled) { settings_.enabled = enabled; }
bool NNBridge::enabled() const { return settings_.enabled; }
void NNBridge::setPython(const std::string& p) { if (!p.empty()) settings_.python = p; }
void NNBridge::setScript(const std::string& s) { if (!s.empty()) settings_.script = s; }
void NNBridge::setModel(const std::string& m) {
    if (!m.empty() && m != settings_.model) {
        shutdown();
        settings_.model = m;
    }
}
void NNBridge::setDevice(const std::string& d) { if (!d.empty()) settings_.device = d; }

NNOutput NNBridge::makeFallbackOutput() const {
    NNOutput out;
    out.value = 0.0;
    out.policy.assign(PolicySize, 1.0 / PolicySize);
    return out;
}

#ifdef _WIN32

bool NNBridge::ensureProcess() {
    if (processRunning_) return true;
    if (!settings_.enabled) return false;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE stdinRead, stdinWrite, stdoutRead, stdoutWrite;
    if (!CreatePipe(&stdinRead, &stdinWrite, &sa, 0)) return false;
    SetHandleInformation(stdinWrite, HANDLE_FLAG_INHERIT, 0);
    if (!CreatePipe(&stdoutRead, &stdoutWrite, &sa, 0)) {
        CloseHandle(stdinRead); CloseHandle(stdinWrite);
        return false;
    }
    SetHandleInformation(stdoutRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = stdinRead;
    si.hStdOutput = stdoutWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    std::string cmd = "\"" + settings_.python + "\" \"" + settings_.script +
                      "\" serve --model \"" + settings_.model +
                      "\" --device \"" + settings_.device + "\"";

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);

    CloseHandle(stdinRead);
    CloseHandle(stdoutWrite);

    if (!ok) {
        CloseHandle(stdinWrite);
        CloseHandle(stdoutRead);
        lastError_ = "failed to start Python process: " + cmd;
        return false;
    }

    CloseHandle(pi.hThread);
    childStdinWrite_ = stdinWrite;
    childStdoutRead_ = stdoutRead;
    processHandle_ = pi.hProcess;
    processRunning_ = true;

    // Wait for "ready" line
    std::string ready = recvLine();
    if (ready.find("ready") == std::string::npos) {
        lastError_ = "Python process did not respond with 'ready' (model load may have failed)";
        shutdown();
        return false;
    }
    lastError_.clear();
    return true;
}

void NNBridge::shutdown() {
    if (!processRunning_) return;
    sendLine("quit");
    WaitForSingleObject(processHandle_, 3000);
    CloseHandle(childStdinWrite_);
    CloseHandle(childStdoutRead_);
    CloseHandle(processHandle_);
    childStdinWrite_ = nullptr;
    childStdoutRead_ = nullptr;
    processHandle_ = nullptr;
    processRunning_ = false;
}

bool NNBridge::sendLine(const std::string& line) {
    std::string data = line + "\n";
    DWORD written;
    return WriteFile(childStdinWrite_, data.c_str(),
                     static_cast<DWORD>(data.size()), &written, nullptr) != 0;
}

std::string NNBridge::recvLine() {
    std::string result;
    char ch;
    DWORD bytesRead;
    while (ReadFile(childStdoutRead_, &ch, 1, &bytesRead, nullptr) && bytesRead > 0) {
        if (ch == '\n') break;
        if (ch != '\r') result += ch;
    }
    return result;
}

#else // POSIX

bool NNBridge::ensureProcess() {
    if (processRunning_) return true;
    if (!settings_.enabled) return false;

    int stdinPipe[2], stdoutPipe[2];
    if (pipe(stdinPipe) != 0 || pipe(stdoutPipe) != 0) {
        lastError_ = "pipe() failed";
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdinPipe[0]); close(stdinPipe[1]);
        close(stdoutPipe[0]); close(stdoutPipe[1]);
        lastError_ = "fork() failed";
        return false;
    }

    if (pid == 0) {
        dup2(stdinPipe[0], STDIN_FILENO);
        dup2(stdoutPipe[1], STDOUT_FILENO);
        close(stdinPipe[0]); close(stdinPipe[1]);
        close(stdoutPipe[0]); close(stdoutPipe[1]);
        execlp(settings_.python.c_str(), settings_.python.c_str(),
               settings_.script.c_str(), "serve",
               "--model", settings_.model.c_str(),
               "--device", settings_.device.c_str(),
               nullptr);
        _exit(1);
    }

    close(stdinPipe[0]);
    close(stdoutPipe[1]);
    childStdinFd_ = stdinPipe[1];
    childStdoutFd_ = stdoutPipe[0];
    childPid_ = pid;
    processRunning_ = true;

    std::string ready = recvLine(30000);
    if (ready.find("ready") == std::string::npos) {
        lastError_ = "Python process did not respond with 'ready' (model load may have failed)";
        shutdown();
        return false;
    }
    lastError_.clear();
    return true;
}

void NNBridge::shutdown() {
    if (!processRunning_) return;
    // Check if child is still alive before writing
    int wstatus;
    pid_t ret = waitpid(childPid_, &wstatus, WNOHANG);
    if (ret == 0) {
        sendLine("quit");
        waitpid(childPid_, &wstatus, 0);
    }
    close(childStdinFd_);
    close(childStdoutFd_);
    childStdinFd_ = -1;
    childStdoutFd_ = -1;
    childPid_ = -1;
    processRunning_ = false;
}

bool NNBridge::sendLine(const std::string& line) {
    std::string data = line + "\n";
    return write(childStdinFd_, data.c_str(), data.size()) > 0;
}

std::string NNBridge::recvLine() {
    return recvLine(-1);
}

std::string NNBridge::recvLine(int timeoutMs) {
    std::string result;
    char ch;
    while (true) {
        if (timeoutMs >= 0) {
            struct pollfd pfd{};
            pfd.fd = childStdoutFd_;
            pfd.events = POLLIN;
            int ret = poll(&pfd, 1, timeoutMs);
            if (ret <= 0) break;
        }
        if (read(childStdoutFd_, &ch, 1) != 1) break;
        if (ch == '\n') break;
        if (ch != '\r') result += ch;
    }
    return result;
}

#endif

// --- Evaluate ---

NNOutput NNBridge::evaluate(const Board& board) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureProcess()) return makeFallbackOutput();

    std::string encoded = encodeBoardState(board);
    if (!sendLine("eval " + encoded)) {
        processRunning_ = false;
        return makeFallbackOutput();
    }

    std::string response = recvLine();
    if (response.empty()) {
        processRunning_ = false;
        return makeFallbackOutput();
    }

    // Parse response: value p0 p1 p2 ... p2186
    NNOutput out;
    std::istringstream iss(response);
    iss >> out.value;
    out.policy.resize(PolicySize);
    for (int i = 0; i < PolicySize; ++i) {
        if (!(iss >> out.policy[i])) {
            out.policy[i] = 1e-6;
        }
    }
    return out;
}

std::vector<NNOutput> NNBridge::evaluateBatch(const std::vector<Board>& boards) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureProcess() || boards.empty()) {
        std::vector<NNOutput> fallback(boards.size());
        for (auto& o : fallback) o = makeFallbackOutput();
        return fallback;
    }

    std::ostringstream cmd;
    cmd << "batch " << boards.size();
    for (const auto& b : boards) {
        cmd << " | " << encodeBoardState(b);
    }
    if (!sendLine(cmd.str())) {
        processRunning_ = false;
        std::vector<NNOutput> fallback(boards.size());
        for (auto& o : fallback) o = makeFallbackOutput();
        return fallback;
    }

    std::vector<NNOutput> results;
    results.reserve(boards.size());
    for (std::size_t i = 0; i < boards.size(); ++i) {
        std::string response = recvLine();
        NNOutput out;
        std::istringstream iss(response);
        iss >> out.value;
        out.policy.resize(PolicySize);
        for (int j = 0; j < PolicySize; ++j) {
            if (!(iss >> out.policy[j])) out.policy[j] = 1e-6;
        }
        results.push_back(std::move(out));
    }
    return results;
}

bool NNBridge::train(const std::string& dataPath, int epochs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureProcess()) return false;

    std::ostringstream cmd;
    cmd << "train " << dataPath << " " << epochs;
    if (!sendLine(cmd.str())) {
        processRunning_ = false;
        return false;
    }

    std::string response = recvLine();
    return response.find("ok") != std::string::npos;
}

} // namespace shogi
