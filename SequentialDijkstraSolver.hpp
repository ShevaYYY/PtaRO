#pragma once

#include "Matrix.hpp"
#include <vector>
#include <queue>
#include <stdexcept>
#include <algorithm>
#include <string>

using namespace std;

class SequentialDijkstraSolver {
public:
    static constexpr double INF = 1e9;

    SequentialDijkstraSolver() = default;

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

        vector<double> dist(N, INF);
        dist[startVertex] = 0.0;

        using Element = pair<double, size_t>;
        priority_queue<Element, vector<Element>, greater<Element>> pq;
        pq.push({0.0, startVertex});

        while (!pq.empty()) {
            auto [currDist, u] = pq.top();
            pq.pop();

            if (currDist > dist[u]) {
                continue;
            }

            for (size_t v = 0; v < N; ++v) {
                if (u == v) {
                    continue;
                }
                double weight = graph(u, v);
                if (weight < INF) {
                    double newDist = currDist + weight;
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                        pq.push({newDist, v});
                    }
                }
            }
        }

        return dist;
    }
};

