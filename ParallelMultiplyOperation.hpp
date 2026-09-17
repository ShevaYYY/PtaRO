#pragma once

#include "IMatrixOperation.hpp"
#include "ThreadChunkDistributor.hpp"
#include <thread>
#include <vector>
#include <stdexcept>
#include <string>

using namespace std;

/**
 * @brief Multi-threaded matrix multiplication using std::thread and 1D Row Chunking.
 * Mutex-free & race-condition-free: Each thread computes a unique slice of rows [startRow, endRow).
 * Cache-optimized loop order: i -> k -> j.
 */
class ParallelMultiplyOperation : public IMatrixOperation {
private:
    size_t requestedThreads_;

public:
    explicit ParallelMultiplyOperation(size_t requestedThreads = thread::hardware_concurrency())
        : requestedThreads_(requestedThreads > 0 ? requestedThreads : 1) {}

    void setRequestedThreads(size_t threads) {
        requestedThreads_ = threads > 0 ? threads : 1;
    }

    size_t getRequestedThreads() const {
        return requestedThreads_;
    }

    Matrix execute(const Matrix& A, const Matrix& B) const override {
        if (A.getCols() != B.getRows()) {
            throw invalid_argument("Matrix dimension mismatch for multiplication! (colsA != rowsB)");
        }

        size_t rowsA = A.getRows();
        size_t colsA = A.getCols();
        size_t colsB = B.getCols();

        Matrix result(rowsA, colsB, 0.0);

        if (rowsA == 0 || colsB == 0) {
            return result;
        }

        // 1D Row Chunking across rows of Matrix A
        auto rowChunks = ThreadChunkDistributor::distribute(rowsA, requestedThreads_);
        size_t numWorkerThreads = rowChunks.size();

        const auto& dataA = A.data();
        const auto& dataB = B.data();
        auto& dataRes = result.data();

        vector<thread> workers;
        workers.reserve(numWorkerThreads);

        for (const auto& chunk : rowChunks) {
            workers.emplace_back([&dataA, &dataB, &dataRes, colsA, colsB, startRow = chunk.start, endRow = chunk.end]() {
                for (size_t i = startRow; i < endRow; ++i) {
                    size_t rowOffsetA = i * colsA;
                    size_t rowOffsetRes = i * colsB;
                    for (size_t k = 0; k < colsA; ++k) {
                        double valA = dataA[rowOffsetA + k];
                        size_t rowOffsetB = k * colsB;
                        for (size_t j = 0; j < colsB; ++j) {
                            dataRes[rowOffsetRes + j] += valA * dataB[rowOffsetB + j];
                        }
                    }
                }
            });
        }

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        return result;
    }

    string getName() const override {
        return "Parallel Multiply (" + to_string(requestedThreads_) + " threads requested, i-k-j)";
    }
};
