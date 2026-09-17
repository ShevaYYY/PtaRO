#pragma once

#include "IMatrixOperation.hpp"
#include "ThreadChunkDistributor.hpp"
#include <thread>
#include <vector>
#include <stdexcept>
#include <string>

using namespace std;


class ParallelAddOperation : public IMatrixOperation {
private:
    size_t requestedThreads_;

public:
    explicit ParallelAddOperation(size_t requestedThreads = thread::hardware_concurrency())
        : requestedThreads_(requestedThreads > 0 ? requestedThreads : 1) {}

    void setRequestedThreads(size_t threads) {
        requestedThreads_ = threads > 0 ? threads : 1;
    }

    size_t getRequestedThreads() const {
        return requestedThreads_;
    }

    Matrix execute(const Matrix& A, const Matrix& B) const override {
        if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
            throw invalid_argument("Matrix dimension mismatch for addition!");
        }

        Matrix result(A.getRows(), A.getCols());
        size_t totalElements = A.getTotalElements();

        if (totalElements == 0) {
            return result;
        }

        auto chunks = ThreadChunkDistributor::distribute(totalElements, requestedThreads_);
        size_t numWorkerThreads = chunks.size();

        const auto& dataA = A.data();
        const auto& dataB = B.data();
        auto& dataRes = result.data();

        vector<thread> workers;
        workers.reserve(numWorkerThreads);

        for (const auto& chunk : chunks) {
            workers.emplace_back([&dataA, &dataB, &dataRes, start = chunk.start, end = chunk.end]() {
                for (size_t i = start; i < end; ++i) {
                    dataRes[i] = dataA[i] + dataB[i];
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
        return "Parallel Add (" + to_string(requestedThreads_) + " threads requested)";
    }
};
