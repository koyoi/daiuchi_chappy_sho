#include "mcts_usi_protocol.h"

#include <filesystem>
#include <string>

#ifndef _WIN32
#include <signal.h>
#endif

int main(int argc, char** argv) {
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        if (executable.has_parent_path()) {
            std::filesystem::current_path(executable.parent_path());
        }
    } catch (...) {
    }

    shogi::mctsUsiLoop();
    return 0;
}
