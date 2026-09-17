#pragma once

#include "Matrix.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "SequentialMultiplyOperation.hpp"
#include "ParallelMultiplyOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "ThreadChunkDistributor.hpp"

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
    /**
     * @brief Generates adaptive thread sampling steps up to maxThreads.
     * Step 1 in zone <= hardware_concurrency * 2.
     * Adaptive steps (4, 8, 20) for larger thread counts.
     */
    static vector<size_t> generateAdaptiveThreadCounts(size_t maxThreads = 240) {
        vector<size_t> threadCounts;
        size_t hw = thread::hardware_concurrency();
        size_t fastZone = hw > 0 ? hw * 2 : 16;

        // Zone 1: Step 1 up to fastZone (e.g., 2 * hardware_concurrency)
        size_t limit1 = min(fastZone, maxThreads);
        for (size_t t = 1; t <= limit1; ++t) {
            threadCounts.push_back(t);
        }

        // Zone 2: Adaptive steps beyond fastZone
        size_t t = limit1;
        while (t < maxThreads) {
            size_t step = 4;
            if (t >= fastZone * 4) {
                step = 20;
            } else if (t >= fastZone * 2) {
                step = 8;
            }

            t += step;
            if (t > maxThreads) {
                t = maxThreads;
            }
            if (threadCounts.back() != t) {
                threadCounts.push_back(t);
            }
        }

        return threadCounts;
    }

    /**
     * @brief Measures execution time of a matrix operation in microseconds.
     */
    static double measureMicroseconds(const IMatrixOperation& op, const Matrix& A, const Matrix& B, int iterations = 3) {
        // Warmup
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

    /**
     * @brief Runs full automated benchmark comparing Baseline vs Parallel ADDITION with Smart Step Sampling.
     */
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

        // 2. Generate Thread Counts with Smart Step Sampling
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

        // Best result detection
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

    /**
     * @brief Runs full automated benchmark comparing Baseline vs Parallel MULTIPLICATION with Smart Step Sampling.
     */
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

        // 1. Measure Baseline Sequential Time
        SequentialMultiplyOperation seqOp;
        double baselineUs = measureMicroseconds(seqOp, A, B, 2);
        double baselineMs = baselineUs / 1000.0;

        cout << " BASELINE (Sequential Single-Threaded Multiply, i-k-j): " 
             << fixed << setprecision(2) << baselineUs << " us (" << baselineMs << " ms)\n";
        cout << "-------------------------------------------------------------------------\n\n";

        // 2. Generate Thread Counts with Smart Step Sampling
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

        // Best result detection
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

    /**
     * @brief Benchmarks Matrix Chain Addition (Binary Tree Task Parallelism vs Sequential Fold).
     */
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

        // 1. Sequential Linear Chain Addition
        auto t1_start = high_resolution_clock::now();
        Matrix resSeq = chainOp.executeSequential(chain, seqSubOp);
        auto t1_end = high_resolution_clock::now();
        double seqUs = duration<double, micro>(t1_end - t1_start).count();

        // 2. Parallel Binary Tree Chain Addition (using Sequential Sub-op)
        auto t2_start = high_resolution_clock::now();
        Matrix resTreeSeq = chainOp.executeParallel(chain, seqSubOp);
        auto t2_end = high_resolution_clock::now();
        double treeSeqUs = duration<double, micro>(t2_end - t2_start).count();

        // 3. Hybrid Parallel Binary Tree Chain Addition (using Multi-threaded Sub-op)
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
};
