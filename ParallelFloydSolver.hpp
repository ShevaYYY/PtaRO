#pragma once

#include "Matrix.hpp"
#include "ThreadChunkDistributor.hpp"
#include <vector>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <string>

using namespace std;

class ParallelFloydSolver {
private:
    size_t requestedThreads_;

public:
    static constexpr double INF = 1e9;

    explicit ParallelFloydSolver(size_t requestedThreads = thread::hardware_concurrency())
        : requestedThreads_(requestedThreads > 0 ? requestedThreads : 1) {}

    void setRequestedThreads(size_t threads) {
        requestedThreads_ = threads > 0 ? threads : 1;
    }

    size_t getRequestedThreads() const {
        return requestedThreads_;
    }

    Matrix solve(const Matrix& graph) const {
        size_t rows = graph.getRows();
        size_t cols = graph.getCols();

        if (rows == 0 || rows != cols) {
            throw invalid_argument("Invalid graph dimensions for Floyd-Warshall! Expected square matrix N x N with N > 0.");
        }

        Matrix dist = graph;
        const size_t N = rows;

        auto chunks = ThreadChunkDistributor::distribute(N, requestedThreads_);

        for (size_t k = 0; k < N; ++k) {
            vector<thread> workers;
            workers.reserve(chunks.size());

            for (const auto& chunk : chunks) {
                workers.emplace_back([&dist, k, N, rowStart = chunk.start, rowEnd = chunk.end]() {
                    for (size_t i = rowStart; i < rowEnd; ++i) {
                        double dist_ik = dist(i, k);
                        if (dist_ik >= INF) {
                            continue;
                        }
                        for (size_t j = 0; j < N; ++j) {
                            double dist_kj = dist(k, j);
                            if (dist_kj < INF) {
                                double newDist = dist_ik + dist_kj;
                                if (newDist < dist(i, j)) {
                                    dist(i, j) = newDist;
                                }
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
        }

        return dist;
    }
};

