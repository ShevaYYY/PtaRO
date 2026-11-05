#pragma once

#include "Matrix.hpp"
#include "ThreadChunkDistributor.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <algorithm>
#include <string>

using namespace std;

class ParallelPrimSolver {
private:
    size_t requestedThreads_;

    class Barrier {
    private:
        mutex mtx_;
        condition_variable cv_;
        size_t threshold_;
        size_t count_;
        size_t generation_;

    public:
        explicit Barrier(size_t count)
            : threshold_(count), count_(count), generation_(0) {}

        void wait() {
            unique_lock<mutex> lock(mtx_);
            size_t gen = generation_;
            if (--count_ == 0) {
                generation_++;
                count_ = threshold_;
                cv_.notify_all();
            } else {
                cv_.wait(lock, [this, gen]() { return gen != generation_; });
            }
        }
    };

public:
    static constexpr double INF = 1e9;

    explicit ParallelPrimSolver(size_t requestedThreads = thread::hardware_concurrency())
        : requestedThreads_(requestedThreads > 0 ? requestedThreads : 1) {}

    void setRequestedThreads(size_t threads) {
        requestedThreads_ = threads > 0 ? threads : 1;
    }

    size_t getRequestedThreads() const {
        return requestedThreads_;
    }

    double solve(const Matrix& graph, size_t startVertex = 0) const {
        size_t rows = graph.getRows();
        size_t cols = graph.getCols();

        if (rows == 0 || rows != cols) {
            throw invalid_argument("Invalid graph dimensions for Prim! Expected square matrix N x N with N > 0.");
        }

        const size_t N = rows;
        if (startVertex >= N) {
            throw out_of_range("Start vertex index out of bounds: " + to_string(startVertex) + 
                               " (Graph size: " + to_string(N) + ")");
        }

        auto chunks = ThreadChunkDistributor::distribute(N, requestedThreads_);
        size_t actualThreads = chunks.size();
        if (actualThreads == 0) {
            actualThreads = 1;
        }

        vector<double> minEdge(N, INF);
        vector<char> inMST(N, 0);
        minEdge[startVertex] = 0.0;

        vector<pair<double, size_t>> localMins(actualThreads, {INF, N});
        size_t globalMinVertex = N;
        double globalMinDist = INF;
        double totalWeight = 0.0;
        bool allDone = false;

        Barrier barrierMin(actualThreads);
        Barrier barrierRelax(actualThreads);

        auto workerFunc = [&](size_t t) {
            const size_t chunkStart = chunks[t].start;
            const size_t chunkEnd = chunks[t].end;

            double myMinDist = INF;
            size_t myMinVertex = N;
            for (size_t v = chunkStart; v < chunkEnd; ++v) {
                if (!inMST[v] && minEdge[v] < myMinDist) {
                    myMinDist = minEdge[v];
                    myMinVertex = v;
                }
            }
            localMins[t] = {myMinDist, myMinVertex};

            while (true) {
                barrierMin.wait();

                if (t == 0) {
                    globalMinDist = INF;
                    globalMinVertex = N;
                    for (size_t i = 0; i < actualThreads; ++i) {
                        if (localMins[i].first < globalMinDist) {
                            globalMinDist = localMins[i].first;
                            globalMinVertex = localMins[i].second;
                        }
                    }

                    if (globalMinDist >= INF || globalMinVertex == N) {
                        allDone = true;
                    } else {
                        inMST[globalMinVertex] = 1;
                        totalWeight += globalMinDist;
                    }
                }

                barrierRelax.wait();

                if (allDone) {
                    break;
                }

                const size_t u = globalMinVertex;

                
                for (size_t v = chunkStart; v < chunkEnd; ++v) {
                    if (!inMST[v]) {
                        double weight = graph(u, v);
                        if (weight < minEdge[v]) {
                            minEdge[v] = weight;
                        }
                    }
                }

                myMinDist = INF;
                myMinVertex = N;
                for (size_t v = chunkStart; v < chunkEnd; ++v) {
                    if (!inMST[v] && minEdge[v] < myMinDist) {
                        myMinDist = minEdge[v];
                        myMinVertex = v;
                    }
                }
                localMins[t] = {myMinDist, myMinVertex};
            }
        };

        vector<thread> workers;
        workers.reserve(actualThreads);
        for (size_t t = 0; t < actualThreads; ++t) {
            workers.emplace_back(workerFunc, t);
        }

        for (auto& w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }

        return totalWeight;
    }
};

