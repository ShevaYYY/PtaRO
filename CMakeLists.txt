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
    SequentialGaussSolver.hpp
    ParallelGaussSolver.hpp
    SequentialFloydSolver.hpp
    ParallelFloydSolver.hpp
    MatrixBenchmarkSuite.hpp
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
    SequentialGaussSolver.hpp
    ParallelGaussSolver.hpp
    SequentialFloydSolver.hpp
    ParallelFloydSolver.hpp
    MatrixBenchmarkSuite.hpp
)

find_package(Threads REQUIRED)
target_link_libraries(matrix_app PRIVATE Threads::Threads)
target_link_libraries(test_runner PRIVATE Threads::Threads)
