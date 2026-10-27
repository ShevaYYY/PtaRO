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

class ParallelDijkstraSolver {
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

    explicit ParallelDijkstraSolver(size_t requestedThreads = thread::hardware_concurrency())
        : requestedThreads_(requestedThreads > 0 ? requestedThreads : 1) {}

    void setRequestedThreads(size_t threads) {
        requestedThreads_ = threads > 0 ? threads : 1;
    }

    size_t getRequestedThreads() const {
        return requestedThreads_;
    }

    vector<double> solve(const Matrix& graph, size_t startVertex) const {
        size_t rows = graph.getRows();
        size_t cols = graph.getCols();

        if (rows == 0 || rows != cols) {
            throw invalid_argument("Invalid graph dimensions for Dijkstra! Expected square matrix N x N with N > 0.");
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

        vector<double> dist(N, INF);
        vector<char> visited(N, 0);
        dist[startVertex] = 0.0;

        vector<pair<double, size_t>> localMins(actualThreads, {INF, N});
        size_t globalMinVertex = N;
        double globalMinDist = INF;
        bool allDone = false;

        Barrier barrierMin(actualThreads);
        Barrier barrierRelax(actualThreads);

        auto workerFunc = [&](size_t t) {
            const size_t chunkStart = chunks[t].start;
            const size_t chunkEnd = chunks[t].end;

            double myMinDist = INF;
            size_t myMinVertex = N;
            for (size_t v = chunkStart; v < chunkEnd; ++v) {
                if (!visited[v] && dist[v] < myMinDist) {
                    myMinDist = dist[v];
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
                        visited[globalMinVertex] = 1;
                    }
                }

                barrierRelax.wait();

                if (allDone) {
                    break;
                }

                const size_t u = globalMinVertex;
                const double uDist = globalMinDist;

                for (size_t v = chunkStart; v < chunkEnd; ++v) {
                    if (u == v) {
                        continue;
                    }
                    if (!visited[v]) {
                        double weight = graph(u, v);
                        if (weight < INF) {
                            double newDist = uDist + weight;
                            if (newDist < dist[v]) {
                                dist[v] = newDist;
                            }
                        }
                    }
                }

                myMinDist = INF;
                myMinVertex = N;
                for (size_t v = chunkStart; v < chunkEnd; ++v) {
                    if (!visited[v] && dist[v] < myMinDist) {
                        myMinDist = dist[v];
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

        return dist;
    }
};

