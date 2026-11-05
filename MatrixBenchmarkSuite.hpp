#pragma once

#include "Matrix.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "SequentialMultiplyOperation.hpp"
#include "ParallelMultiplyOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "ThreadChunkDistributor.hpp"
#include "SequentialGaussSolver.hpp"
#include "ParallelGaussSolver.hpp"
#include "SequentialFloydSolver.hpp"
#include "ParallelFloydSolver.hpp"
#include "SequentialDijkstraSolver.hpp"
#include "ParallelDijkstraSolver.hpp"
#include "SequentialPrimSolver.hpp"
#include "ParallelPrimSolver.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

using namespace std;
using namespace std::chrono;

struct BenchmarkResult {
    size_t requestedThreads;
    size_t actualThreads;
    double timeUs;
    double timeMs;
    double speedup;
    double efficiency;
};

class MatrixBenchmarkSuite {
public:

    static vector<size_t> generateAdaptiveThreadCounts(size_t maxThreads = 240) {
        vector<size_t> baseSteps = {1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 100};
        vector<size_t> result;

        for (size_t threads : baseSteps) {
            if (threads <= maxThreads) {
                result.push_back(threads);
            }
        }

        if (!result.empty() && result.back() < maxThreads) {
            result.push_back(maxThreads);
        }

        return result;
    }

    static double measureMicroseconds(const IMatrixOperation& op, const Matrix& A, const Matrix& B, int iterations = 3) {
        op.execute(A, B);

        double totalUs = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            Matrix res = op.execute(A, B);
            auto end = high_resolution_clock::now();
            totalUs += duration<double, micro>(end - start).count();
        }
        return totalUs / iterations;
    }


    static void runAutoBenchmark(const Matrix& A, const Matrix& B, size_t maxThreads = 240) {
        cout << "\n=========================================================================\n";
        cout << "                  AUTOMATED MATRIX ADDITION BENCHMARK                   \n";
        cout << "=========================================================================\n";
        cout << " Matrix Size: " << A.getRows() << " x " << A.getCols() 
             << " (" << A.getTotalElements() << " elements)\n";
        cout << " Hardware Concurrency (Logical Cores): " << thread::hardware_concurrency() << "\n";
        cout << " Target Max Threads for Sampling: " << maxThreads << "\n";
        cout << "-------------------------------------------------------------------------\n";

        // 1. Measure Baseline Sequential Time
        SequentialAddOperation seqOp;
        double baselineUs = measureMicroseconds(seqOp, A, B, 3);
        double baselineMs = baselineUs / 1000.0;

        cout << " BASELINE (Sequential Single-Threaded Add): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << "-------------------------------------------------------------------------\n\n";

        vector<size_t> sampleCounts = generateAdaptiveThreadCounts(maxThreads);

        cout << left 
             << setw(12) << "| Threads" 
             << setw(16) << "| Actual Threads" 
             << setw(14) << "| Time (us)" 
             << setw(14) << "| Time (ms)" 
             << setw(14) << "| Speedup (x)" 
             << setw(16) << "| Efficiency (%)" << "|\n";
        cout << "|------------|----------------|--------------|--------------|--------------|----------------|\n";

        vector<BenchmarkResult> results;

        for (size_t reqThreads : sampleCounts) {
            ParallelAddOperation parOp(reqThreads);
            double parUs = measureMicroseconds(parOp, A, B, 3);
            double parMs = parUs / 1000.0;

            size_t actualThreads = min(A.getTotalElements(), reqThreads);
            if (actualThreads == 0 && A.getTotalElements() > 0) actualThreads = 1;

            double speedup = baselineUs / parUs;
            double efficiency = (speedup / reqThreads) * 100.0;

            results.push_back({ reqThreads, actualThreads, parUs, parMs, speedup, efficiency });

            cout << left 
                 << "| " << setw(11) << reqThreads 
                 << "| " << setw(15) << actualThreads 
                 << "| " << setw(13) << fixed << setprecision(2) << parUs 
                 << "| " << setw(13) << fixed << setprecision(3) << parMs 
                 << "| " << setw(13) << fixed << setprecision(2) << speedup 
                 << "| " << setw(15) << fixed << setprecision(1) << efficiency << "|\n";
        }
        cout << "-------------------------------------------------------------------------\n";

        auto bestIt = max_element(results.begin(), results.end(), [](const BenchmarkResult& r1, const BenchmarkResult& r2) {
            return r1.speedup < r2.speedup;
        });

        if (bestIt != results.end()) {
            cout << "\n>>> BEST ADDITION PERFORMANCE RESULT <<<\n";
            cout << " Optimal Thread Count : " << bestIt->requestedThreads << " threads\n";
            cout << " Minimum Execution Time: " << bestIt->timeUs << " us (" << bestIt->timeMs << " ms)\n";
            cout << " Maximum Speedup       : " << fixed << setprecision(2) << bestIt->speedup << "x relative to Baseline\n";
        }
        cout << "=========================================================================\n\n";
    }

    static void runMultiplyBenchmark(const Matrix& A, const Matrix& B, size_t maxThreads = 240) {
        if (A.getCols() != B.getRows()) {
            cout << "[Error] Matrix dimensions incompatible for multiplication! (colsA != rowsB)\n";
            return;
        }

        cout << "\n=========================================================================\n";
        cout << "               AUTOMATED MATRIX MULTIPLICATION BENCHMARK                 \n";
        cout << "=========================================================================\n";
        cout << " Matrix A: " << A.getRows() << " x " << A.getCols() << "\n";
        cout << " Matrix B: " << B.getRows() << " x " << B.getCols() << "\n";
        cout << " Result Matrix C: " << A.getRows() << " x " << B.getCols() 
             << " (" << (A.getRows() * B.getCols()) << " elements)\n";
        cout << " Hardware Concurrency (Logical Cores): " << thread::hardware_concurrency() << "\n";
        cout << " Target Max Threads for Sampling: " << maxThreads << "\n";
        cout << " Optimization Strategy: Cache-optimized i-k-j loop order + Row Chunking\n";
        cout << "-------------------------------------------------------------------------\n";

        SequentialMultiplyOperation seqOp;
        double baselineUs = measureMicroseconds(seqOp, A, B, 2);
        double baselineMs = baselineUs / 1000.0;

        cout << " BASELINE (Sequential Single-Threaded Multiply, i-k-j): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << "-------------------------------------------------------------------------\n\n";

        vector<size_t> sampleCounts = generateAdaptiveThreadCounts(maxThreads);

        cout << left 
             << setw(12) << "| Threads" 
             << setw(16) << "| Actual Threads" 
             << setw(14) << "| Time (us)" 
             << setw(14) << "| Time (ms)" 
             << setw(14) << "| Speedup (x)" 
             << setw(16) << "| Efficiency (%)" << "|\n";
        cout << "|------------|----------------|--------------|--------------|--------------|----------------|\n";

        vector<BenchmarkResult> results;

        for (size_t reqThreads : sampleCounts) {
            ParallelMultiplyOperation parOp(reqThreads);
            double parUs = measureMicroseconds(parOp, A, B, 2);
            double parMs = parUs / 1000.0;

            size_t actualThreads = min(A.getRows(), reqThreads);
            if (actualThreads == 0 && A.getRows() > 0) actualThreads = 1;

            double speedup = baselineUs / parUs;
            double efficiency = (speedup / reqThreads) * 100.0;

            results.push_back({ reqThreads, actualThreads, parUs, parMs, speedup, efficiency });

            cout << left 
                 << "| " << setw(11) << reqThreads 
                 << "| " << setw(15) << actualThreads 
                 << "| " << setw(13) << fixed << setprecision(2) << parUs 
                 << "| " << setw(13) << fixed << setprecision(3) << parMs 
                 << "| " << setw(13) << fixed << setprecision(2) << speedup 
                 << "| " << setw(15) << fixed << setprecision(1) << efficiency << "|\n";
        }
        cout << "-------------------------------------------------------------------------\n";

        auto bestIt = max_element(results.begin(), results.end(), [](const BenchmarkResult& r1, const BenchmarkResult& r2) {
            return r1.speedup < r2.speedup;
        });

        if (bestIt != results.end()) {
            cout << "\n>>> BEST MULTIPLICATION PERFORMANCE RESULT <<<\n";
            cout << " Optimal Thread Count : " << bestIt->requestedThreads << " threads\n";
            cout << " Minimum Execution Time: " << bestIt->timeUs << " us (" << bestIt->timeMs << " ms)\n";
            cout << " Maximum Speedup       : " << fixed << setprecision(2) << bestIt->speedup << "x relative to Baseline\n";
        }
        cout << "=========================================================================\n\n";
    }


    static void runChainBenchmark(const vector<Matrix>& chain) {
        if (chain.size() < 2) {
            cout << "[Warning] Matrix chain must contain at least 2 matrices for benchmark.\n";
            return;
        }

        cout << "\n=========================================================================\n";
        cout << "                 CHAIN MATRIX ADDITION BENCHMARK                        \n";
        cout << "=========================================================================\n";
        cout << " Matrix Count in Chain: " << chain.size() << "\n";
        cout << " Dimensions per Matrix: " << chain[0].getRows() << " x " << chain[0].getCols() << "\n";
        cout << "-------------------------------------------------------------------------\n";

        ChainMatrixAddOperation chainOp;
        SequentialAddOperation seqSubOp;
        ParallelAddOperation parSubOp(thread::hardware_concurrency());

        auto t1_start = high_resolution_clock::now();
        Matrix resSeq = chainOp.executeSequential(chain, seqSubOp);
        auto t1_end = high_resolution_clock::now();
        double seqUs = duration<double, micro>(t1_end - t1_start).count();

        auto t2_start = high_resolution_clock::now();
        Matrix resTreeSeq = chainOp.executeParallel(chain, seqSubOp);
        auto t2_end = high_resolution_clock::now();
        double treeSeqUs = duration<double, micro>(t2_end - t2_start).count();

        auto t3_start = high_resolution_clock::now();
        Matrix resTreePar = chainOp.executeParallel(chain, parSubOp);
        auto t3_end = high_resolution_clock::now();
        double treeParUs = duration<double, micro>(t3_end - t3_start).count();

        cout << left
             << setw(45) << " Method"
             << setw(15) << " Time (us)"
             << setw(15) << " Time (ms)"
             << setw(15) << " Speedup (x)" << "\n";
        cout << "-------------------------------------------------------------------------\n";

        cout << left 
             << setw(45) << " 1. Sequential Linear Chain"
             << setw(15) << fixed << setprecision(2) << seqUs
             << setw(15) << fixed << setprecision(3) << (seqUs / 1000.0)
             << setw(15) << "1.00x (Baseline)\n";

        cout << left 
             << setw(45) << " 2. Binary Tree Task Parallelism (async)"
             << setw(15) << fixed << setprecision(2) << treeSeqUs
             << setw(15) << fixed << setprecision(3) << (treeSeqUs / 1000.0)
             << setw(15) << fixed << setprecision(2) << (seqUs / treeSeqUs) << "x\n";

        cout << left 
             << setw(45) << " 3. Binary Tree + Threaded Sub-Ops (Hybrid)"
             << setw(15) << fixed << setprecision(2) << treeParUs
             << setw(15) << fixed << setprecision(3) << (treeParUs / 1000.0)
             << setw(15) << fixed << setprecision(2) << (seqUs / treeParUs) << "x\n";

        cout << "=========================================================================\n\n";
    }

    static Matrix generateWellConditionedSLAE(size_t N) {
        Matrix augmented(N, N + 1);
        augmented.randomize(1.0, 50.0);

        for (size_t i = 0; i < N; ++i) {
            augmented(i, i) += static_cast<double>(N) * 10.0;
        }

        return augmented;
    }


    template <typename SolverType>
    static double measureGaussMicroseconds(const SolverType& solver, const Matrix& augmented, int iterations = 2) {
        solver.solve(augmented);

        double totalUs = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            auto res = solver.solve(augmented);
            auto end = high_resolution_clock::now();
            totalUs += duration<double, micro>(end - start).count();
        }
        return totalUs / iterations;
    }

    static void runGaussBenchmark(size_t systemSize, size_t maxThreads = 240) {
        if (systemSize == 0) {
            cout << "[Error] System size N must be > 0!\n";
            return;
        }

        cout << "\n=========================================================================\n";
        cout << "               AUTOMATED SLAE GAUSS ELIMINATION BENCHMARK                \n";
        cout << "=========================================================================\n";
        cout << " System Size (Equations / Variables N): " << systemSize << "\n";
        cout << " Augmented Matrix Size: " << systemSize << " x " << (systemSize + 1) 
             << " (" << (systemSize * (systemSize + 1)) << " elements)\n";
        cout << " Hardware Concurrency (Logical Cores): " << thread::hardware_concurrency() << "\n";
        cout << " Target Max Threads for Sampling: " << maxThreads << "\n";
        cout << " Elimination Strategy: Per-step Row Partitioning (Join Barrier)\n";
        cout << "-------------------------------------------------------------------------\n";

        Matrix augmented = generateWellConditionedSLAE(systemSize);

        SequentialGaussSolver seqSolver;
        double baselineUs = measureGaussMicroseconds(seqSolver, augmented, 2);
        double baselineMs = baselineUs / 1000.0;

        cout << " BASELINE (Sequential Single-Threaded Gauss): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << "-------------------------------------------------------------------------\n\n";

        vector<size_t> sampleCounts = generateAdaptiveThreadCounts(maxThreads);

        cout << left 
             << setw(12) << "| Threads" 
             << setw(16) << "| Actual Threads" 
             << setw(14) << "| Time (us)" 
             << setw(14) << "| Time (ms)" 
             << setw(14) << "| Speedup (x)" 
             << setw(16) << "| Efficiency (%)" << "|\n";
        cout << "|------------|----------------|--------------|--------------|--------------|----------------|\n";

        vector<BenchmarkResult> results;

        for (size_t reqThreads : sampleCounts) {
            ParallelGaussSolver parSolver(reqThreads);
            double parUs = measureGaussMicroseconds(parSolver, augmented, 2);
            double parMs = parUs / 1000.0;

            size_t actualThreads = min(systemSize, reqThreads);
            if (actualThreads == 0 && systemSize > 0) actualThreads = 1;

            double speedup = baselineUs / parUs;
            double efficiency = (speedup / reqThreads) * 100.0;

            results.push_back({ reqThreads, actualThreads, parUs, parMs, speedup, efficiency });

            cout << left 
                 << "| " << setw(11) << reqThreads 
                 << "| " << setw(15) << actualThreads 
                 << "| " << setw(13) << fixed << setprecision(2) << parUs 
                 << "| " << setw(13) << fixed << setprecision(3) << parMs 
                 << "| " << setw(13) << fixed << setprecision(2) << speedup 
                 << "| " << setw(15) << fixed << setprecision(1) << efficiency << "|\n";
        }
        cout << "-------------------------------------------------------------------------\n";

        auto bestIt = max_element(results.begin(), results.end(), [](const BenchmarkResult& r1, const BenchmarkResult& r2) {
            return r1.speedup < r2.speedup;
        });

        if (bestIt != results.end()) {
            cout << "\n>>> BEST GAUSS ELIMINATION PERFORMANCE RESULT <<<\n";
            cout << " Optimal Thread Count : " << bestIt->requestedThreads << " threads\n";
            cout << " Minimum Execution Time: " << bestIt->timeUs << " us (" << bestIt->timeMs << " ms)\n";
            cout << " Maximum Speedup       : " << fixed << setprecision(2) << bestIt->speedup << "x relative to Baseline\n";
        }
        cout << "=========================================================================\n\n";
    }

    static Matrix generateRandomGraph(size_t N, double edgeProbability = 0.7) {
        Matrix D(N, N, SequentialFloydSolver::INF);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> probDis(0.0, 1.0);
        uniform_real_distribution<double> weightDis(1.0, 100.0);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i == j) {
                    D(i, j) = 0.0;
                } else if (probDis(gen) < edgeProbability) {
                    D(i, j) = weightDis(gen);
                } else {
                    D(i, j) = SequentialFloydSolver::INF;
                }
            }
        }
        return D;
    }

    template <typename SolverType>
    static double measureFloydMicroseconds(const SolverType& solver, const Matrix& graph, int iterations = 1) {
        solver.solve(graph);

        double totalUs = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            auto res = solver.solve(graph);
            auto end = high_resolution_clock::now();
            totalUs += duration<double, micro>(end - start).count();
        }
        return totalUs / iterations;
    }

    static void runFloydBenchmark(size_t graphSize, size_t maxThreads = 240) {
        if (graphSize == 0) {
            cout << "[Error] Graph size N must be > 0!\n";
            return;
        }

        cout << "\n=========================================================================\n";
        cout << "            AUTOMATED FLOYD-WARSHALL SHORTEST PATH BENCHMARK             \n";
        cout << "=========================================================================\n";
        cout << " Vertices Count (N): " << graphSize << "\n";
        cout << " Distance Matrix Size: " << graphSize << " x " << graphSize 
             << " (" << (graphSize * graphSize) << " elements)\n";
        cout << " Hardware Concurrency (Logical Cores): " << thread::hardware_concurrency() << "\n";
        cout << " Target Max Threads for Sampling: " << maxThreads << "\n";
        cout << " Parallelization Strategy: Outer k Barrier + Middle Row Chunking (join)\n";
        cout << "-------------------------------------------------------------------------\n";

        Matrix graph = generateRandomGraph(graphSize, 0.7);

        int iterations = (graphSize <= 250) ? 2 : 1;

        SequentialFloydSolver seqSolver;
        double baselineUs = measureFloydMicroseconds(seqSolver, graph, iterations);
        double baselineMs = baselineUs / 1000.0;

        cout << " BASELINE (Sequential Single-Threaded Floyd): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << "-------------------------------------------------------------------------\n\n";

        vector<size_t> sampleCounts = generateAdaptiveThreadCounts(maxThreads);

        cout << left 
             << setw(12) << "| Threads" 
             << setw(16) << "| Actual Threads" 
             << setw(14) << "| Time (us)" 
             << setw(14) << "| Time (ms)" 
             << setw(14) << "| Speedup (x)" 
             << setw(16) << "| Efficiency (%)" << "|\n";
        cout << "|------------|----------------|--------------|--------------|--------------|----------------|\n";

        vector<BenchmarkResult> results;

        for (size_t reqThreads : sampleCounts) {
            ParallelFloydSolver parSolver(reqThreads);
            double parUs = measureFloydMicroseconds(parSolver, graph, iterations);
            double parMs = parUs / 1000.0;

            size_t actualThreads = min(graphSize, reqThreads);
            if (actualThreads == 0 && graphSize > 0) actualThreads = 1;

            double speedup = baselineUs / parUs;
            double efficiency = (speedup / reqThreads) * 100.0;

            results.push_back({ reqThreads, actualThreads, parUs, parMs, speedup, efficiency });

            cout << left 
                 << "| " << setw(11) << reqThreads 
                 << "| " << setw(15) << actualThreads 
                 << "| " << setw(13) << fixed << setprecision(2) << parUs 
                 << "| " << setw(13) << fixed << setprecision(3) << parMs 
                 << "| " << setw(13) << fixed << setprecision(2) << speedup 
                 << "| " << setw(15) << fixed << setprecision(1) << efficiency << "|\n";
        }
        cout << "-------------------------------------------------------------------------\n";

        auto bestIt = max_element(results.begin(), results.end(), [](const BenchmarkResult& r1, const BenchmarkResult& r2) {
            return r1.speedup < r2.speedup;
        });

        if (bestIt != results.end()) {
            cout << "\n>>> BEST FLOYD-WARSHALL PERFORMANCE RESULT <<<\n";
            cout << " Optimal Thread Count : " << bestIt->requestedThreads << " threads\n";
            cout << " Minimum Execution Time: " << bestIt->timeUs << " us (" << bestIt->timeMs << " ms)\n";
            cout << " Maximum Speedup       : " << fixed << setprecision(2) << bestIt->speedup << "x relative to Baseline\n";
        }
        cout << "=========================================================================\n\n";
    }

    template <typename SolverType>
    static double measureDijkstraMicroseconds(const SolverType& solver, const Matrix& graph, size_t startVertex, int iterations = 3) {
        solver.solve(graph, startVertex);

        double totalUs = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            solver.solve(graph, startVertex);
            auto end = high_resolution_clock::now();
            totalUs += duration<double, micro>(end - start).count();
        }
        return totalUs / iterations;
    }

    static void runDijkstraBenchmark(size_t graphSize, size_t startVertex = 0, size_t maxThreads = 240) {
        if (graphSize == 0) {
            cout << "[Error] Graph size N must be > 0!\n";
            return;
        }
        if (startVertex >= graphSize) {
            cout << "[Error] Start vertex must be < " << graphSize << "!\n";
            return;
        }

        cout << "\n=========================================================================\n";
        cout << "            AUTOMATED DIJKSTRA SHORTEST PATH BENCHMARK                   \n";
        cout << "=========================================================================\n";
        cout << " Vertices Count (N): " << graphSize << "\n";
        cout << " Start Vertex: " << startVertex << "\n";
        cout << " Adjacency Matrix Size: " << graphSize << " x " << graphSize 
             << " (" << (graphSize * graphSize) << " elements)\n";
        cout << " Hardware Concurrency (Logical Cores): " << thread::hardware_concurrency() << "\n";
        cout << " Target Max Threads for Sampling: " << maxThreads << "\n";
        cout << " Parallelization Strategy: Barrier-Synchronized 1D Vertex Chunk Partitioning\n";
        cout << "-------------------------------------------------------------------------\n";

        Matrix graph = generateRandomGraph(graphSize, 0.7);

        int iterations = (graphSize <= 500) ? 5 : 2;

        SequentialDijkstraSolver seqSolver;
        double baselineUs = measureDijkstraMicroseconds(seqSolver, graph, startVertex, iterations);
        double baselineMs = baselineUs / 1000.0;

        cout << " BASELINE (Sequential Priority-Queue Dijkstra): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << "-------------------------------------------------------------------------\n\n";

        vector<size_t> sampleCounts = generateAdaptiveThreadCounts(maxThreads);

        cout << left 
             << setw(12) << "| Threads" 
             << setw(16) << "| Actual Threads" 
             << setw(14) << "| Time (us)" 
             << setw(14) << "| Time (ms)" 
             << setw(14) << "| Speedup (x)" 
             << setw(16) << "| Efficiency (%)" << "|\n";
        cout << "|------------|----------------|--------------|--------------|--------------|----------------|\n";

        vector<BenchmarkResult> results;

        for (size_t reqThreads : sampleCounts) {
            ParallelDijkstraSolver parSolver(reqThreads);
            double parUs = measureDijkstraMicroseconds(parSolver, graph, startVertex, iterations);
            double parMs = parUs / 1000.0;

            size_t actualThreads = min(graphSize, reqThreads);
            if (actualThreads == 0 && graphSize > 0) actualThreads = 1;

            double speedup = baselineUs / parUs;
            double efficiency = (speedup / reqThreads) * 100.0;

            results.push_back({ reqThreads, actualThreads, parUs, parMs, speedup, efficiency });

            cout << left 
                 << "| " << setw(11) << reqThreads 
                 << "| " << setw(15) << actualThreads 
                 << "| " << setw(13) << fixed << setprecision(2) << parUs 
                 << "| " << setw(13) << fixed << setprecision(3) << parMs 
                 << "| " << setw(13) << fixed << setprecision(2) << speedup 
                 << "| " << setw(15) << fixed << setprecision(1) << efficiency << "|\n";
        }
        cout << "-------------------------------------------------------------------------\n";

        auto bestIt = max_element(results.begin(), results.end(), [](const BenchmarkResult& r1, const BenchmarkResult& r2) {
            return r1.speedup < r2.speedup;
        });

        if (bestIt != results.end()) {
            cout << "\n>>> BEST DIJKSTRA PERFORMANCE RESULT <<<\n";
            cout << " Optimal Thread Count : " << bestIt->requestedThreads << " threads\n";
            cout << " Minimum Execution Time: " << bestIt->timeUs << " us (" << bestIt->timeMs << " ms)\n";
            cout << " Maximum Speedup       : " << fixed << setprecision(2) << bestIt->speedup << "x relative to Baseline\n";
        }
        cout << "=========================================================================\n\n";
    }

    static Matrix generateRandomSymmetricGraph(size_t N, double edgeProbability = 0.7) {
        Matrix D(N, N, SequentialPrimSolver::INF);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> probDis(0.0, 1.0);
        uniform_real_distribution<double> weightDis(1.0, 100.0);

        for (size_t i = 0; i < N; ++i) {
            D(i, i) = 0.0;
            for (size_t j = i + 1; j < N; ++j) {
                if (probDis(gen) < edgeProbability) {
                    double w = weightDis(gen);
                    D(i, j) = w;
                    D(j, i) = w;
                } else {
                    D(i, j) = SequentialPrimSolver::INF;
                    D(j, i) = SequentialPrimSolver::INF;
                }
            }
        }
        return D;
    }

    template <typename SolverType>
    static double measurePrimMicroseconds(const SolverType& solver, const Matrix& graph, size_t startVertex = 0, int iterations = 3) {
        solver.solve(graph, startVertex);

        double totalUs = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            solver.solve(graph, startVertex);
            auto end = high_resolution_clock::now();
            totalUs += duration<double, micro>(end - start).count();
        }
        return totalUs / iterations;
    }

    static void runPrimBenchmark(size_t graphSize, size_t startVertex = 0, size_t maxThreads = 240) {
        if (graphSize == 0) {
            cout << "[Error] Graph size N must be > 0!\n";
            return;
        }
        if (startVertex >= graphSize) {
            cout << "[Error] Start vertex must be < " << graphSize << "!\n";
            return;
        }

        cout << "\n=========================================================================\n";
        cout << "            AUTOMATED PRIM'S MST ALGORITHM BENCHMARK                     \n";
        cout << "=========================================================================\n";
        cout << " Vertices Count (N): " << graphSize << "\n";
        cout << " Start Vertex: " << startVertex << "\n";
        cout << " Symmetric Adjacency Matrix: " << graphSize << " x " << graphSize 
             << " (" << (graphSize * graphSize) << " elements)\n";
        cout << " Hardware Concurrency (Logical Cores): " << thread::hardware_concurrency() << "\n";
        cout << " Target Max Threads for Sampling: " << maxThreads << "\n";
        cout << " Parallelization Strategy: Barrier-Synchronized 1D Vertex Chunk Partitioning\n";
        cout << "-------------------------------------------------------------------------\n";

        Matrix graph = generateRandomSymmetricGraph(graphSize, 0.7);

        int iterations = (graphSize <= 500) ? 5 : 2;

        SequentialPrimSolver seqSolver;
        double baselineUs = measurePrimMicroseconds(seqSolver, graph, startVertex, iterations);
        double baselineMs = baselineUs / 1000.0;
        double baselineMST = seqSolver.solve(graph, startVertex);

        cout << " BASELINE (Sequential Prim's Algorithm): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << " MST Total Weight: " << fixed << setprecision(2) << baselineMST << "\n";
        cout << "-------------------------------------------------------------------------\n\n";

        vector<size_t> sampleCounts = generateAdaptiveThreadCounts(maxThreads);

        cout << left 
             << setw(12) << "| Threads" 
             << setw(16) << "| Actual Threads" 
             << setw(14) << "| Time (us)" 
             << setw(14) << "| Time (ms)" 
             << setw(14) << "| Speedup (x)" 
             << setw(16) << "| Efficiency (%)" << "|\n";
        cout << "|------------|----------------|--------------|--------------|--------------|----------------|\n";

        vector<BenchmarkResult> results;

        for (size_t reqThreads : sampleCounts) {
            ParallelPrimSolver parSolver(reqThreads);
            double parUs = measurePrimMicroseconds(parSolver, graph, startVertex, iterations);
            double parMs = parUs / 1000.0;

            size_t actualThreads = min(graphSize, reqThreads);
            if (actualThreads == 0 && graphSize > 0) actualThreads = 1;

            double speedup = baselineUs / parUs;
            double efficiency = (speedup / reqThreads) * 100.0;

            results.push_back({ reqThreads, actualThreads, parUs, parMs, speedup, efficiency });

            cout << left 
                 << "| " << setw(11) << reqThreads 
                 << "| " << setw(15) << actualThreads 
                 << "| " << setw(13) << fixed << setprecision(2) << parUs 
                 << "| " << setw(13) << fixed << setprecision(3) << parMs 
                 << "| " << setw(13) << fixed << setprecision(2) << speedup 
                 << "| " << setw(15) << fixed << setprecision(1) << efficiency << "|\n";
        }
        cout << "-------------------------------------------------------------------------\n";

        auto bestIt = max_element(results.begin(), results.end(), [](const BenchmarkResult& r1, const BenchmarkResult& r2) {
            return r1.speedup < r2.speedup;
        });

        if (bestIt != results.end()) {
            cout << "\n>>> BEST PRIM PERFORMANCE RESULT <<<\n";
            cout << " Optimal Thread Count : " << bestIt->requestedThreads << " threads\n";
            cout << " Minimum Execution Time: " << bestIt->timeUs << " us (" << bestIt->timeMs << " ms)\n";
            cout << " Maximum Speedup       : " << fixed << setprecision(2) << bestIt->speedup << "x relative to Baseline\n";
        }
        cout << "=========================================================================\n\n";
    }
};
