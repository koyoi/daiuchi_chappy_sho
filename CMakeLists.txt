cmake_minimum_required(VERSION 3.16)

project(random_shogi_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(random-shogi-engine
    src/csa_protocol.cpp
    src/engine.cpp
    src/evaluation.cpp
    src/gpu_bridge.cpp
    src/learning.cpp
    src/main.cpp
    src/movegen.cpp
    src/notation.cpp
    src/position.cpp
    src/shogi_types.cpp
    src/text_util.cpp
    src/usi_protocol.cpp
)

target_compile_features(random-shogi-engine PRIVATE cxx_std_17)

add_custom_command(TARGET random-shogi-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:random-shogi-engine>/tools"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_eval.py"
        "$<TARGET_FILE_DIR:random-shogi-engine>/tools/gpu_eval.py"
)

if(MSVC)
    target_compile_options(random-shogi-engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(random-shogi-engine PRIVATE -Wall -Wextra -Wpedantic)
endif()
