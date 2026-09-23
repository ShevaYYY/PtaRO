cmake_minimum_required(VERSION 3.14)
project(MatrixAdditionBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for performance
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -pthread)
endif()

add_executable(matrix_app
    1_2_3.cpp
    Matrix.hpp
    ThreadChunkDistributor.hpp
    IMatrixOperation.hpp
    SequentialAddOperation.hpp
    ParallelAddOperation.hpp
    SequentialMultiplyOperation.hpp
    ParallelMultiplyOperation.hpp
    ChainMatrixAddOperation.hpp
    MatrixBenchmarkSuite.hpp
    include/SequentialGaussSolver.hpp
    src/SequentialGaussSolver.cpp
    include/ParallelGaussSolver.hpp
    src/ParallelGaussSolver.cpp
)

add_executable(test_runner
    test_runner.cpp
    Matrix.hpp
    ThreadChunkDistributor.hpp
    IMatrixOperation.hpp
    SequentialAddOperation.hpp
    ParallelAddOperation.hpp
    SequentialMultiplyOperation.hpp
    ParallelMultiplyOperation.hpp
    ChainMatrixAddOperation.hpp
    MatrixBenchmarkSuite.hpp
    include/SequentialGaussSolver.hpp
    src/SequentialGaussSolver.cpp
    include/ParallelGaussSolver.hpp
    src/ParallelGaussSolver.cpp
)

target_include_directories(matrix_app PRIVATE include .)
target_include_directories(test_runner PRIVATE include .)

find_package(Threads REQUIRED)
target_link_libraries(matrix_app PRIVATE Threads::Threads)
target_link_libraries(test_runner PRIVATE Threads::Threads)
