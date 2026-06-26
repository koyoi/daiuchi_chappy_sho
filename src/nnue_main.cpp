#include "nnue_usi_protocol.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

int main(int /*argc*/, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stdin, nullptr, _IONBF, 0);
    std::ios::sync_with_stdio(true);
    std::cout << std::unitbuf;

    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        if (executable.has_parent_path())
            std::filesystem::current_path(executable.parent_path());
    } catch (...) {
    }

    shogi::nnueUsiLoop();
    return 0;
}
