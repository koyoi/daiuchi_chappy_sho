cmake_minimum_required(VERSION 3.16)

project(kishi_to LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release: 速度最優先の最適化 (古いCPU非対応)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /GL /GS- /fp:fast /arch:AVX2 /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -ffast-math -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# --- 共通ライブラリ (盤面・指し手生成・局面操作・表記) ---
add_library(shogi_core STATIC
    src/shogi_types.cpp
    src/movegen.cpp
    src/position.cpp
    src/notation.cpp
    src/text_util.cpp
)
target_compile_features(shogi_core PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(shogi_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(shogi_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- 従来エンジン (alpha-beta + 98特徴量 + オンライン学習) ---
add_executable(kishi-to-classic
    src/engine.cpp
    src/evaluation.cpp
    src/learning.cpp
    src/kifu_learner.cpp
    src/mate_solver.cpp
    src/opening_book.cpp
    src/usi_protocol.cpp
    src/csa_protocol.cpp
    src/main.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(kishi-to-classic PRIVATE shogi_core Threads::Threads)
target_compile_features(kishi-to-classic PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(kishi-to-classic PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(kishi-to-classic PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- MCTSエンジン (Transformer + MCTS) ---
option(USE_ONNXRUNTIME "Use ONNX Runtime for direct C++ NN inference" OFF)

set(MCTS_SOURCES
    src/mcts.cpp
    src/nn_bridge.cpp
    src/mcts_engine.cpp
    src/mcts_usi_protocol.cpp
    src/mcts_main.cpp
)

if(USE_ONNXRUNTIME)
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" ENV ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include" ENV ONNXRUNTIME_ROOT
        PATH_SUFFIXES onnxruntime)
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
        list(APPEND MCTS_SOURCES src/onnx_inference.cpp)
    else()
        message(WARNING "ONNX Runtime not found (set ONNXRUNTIME_ROOT). Building without ONNX.")
        set(USE_ONNXRUNTIME OFF)
    endif()
endif()

add_executable(kishi-to ${MCTS_SOURCES})
target_link_libraries(kishi-to PRIVATE shogi_core)
target_compile_features(kishi-to PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(kishi-to PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(kishi-to PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(USE_ONNXRUNTIME)
    target_compile_definitions(kishi-to PRIVATE HAS_ONNXRUNTIME=1)
    target_include_directories(kishi-to PRIVATE ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(kishi-to PRIVATE ${ONNXRUNTIME_LIB})
endif()

add_custom_command(TARGET kishi-to POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kishi-to>/tools"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/nn_eval.py"
        "$<TARGET_FILE_DIR:kishi-to>/tools/nn_eval.py"
)
