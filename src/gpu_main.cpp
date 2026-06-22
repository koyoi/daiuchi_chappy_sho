#include "gpu_usi_protocol.h"

#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        if (executable.has_parent_path()) {
            std::filesystem::current_path(executable.parent_path());
        }
    } catch (...) {
    }

    shogi::gpuUsiLoop();
    return 0;
}
