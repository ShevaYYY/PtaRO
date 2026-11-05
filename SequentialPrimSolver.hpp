#pragma once

#include "Matrix.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>

using namespace std;

class SequentialPrimSolver {
public:
    static constexpr double INF = 1e9;

    SequentialPrimSolver() = default;

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

        vector<double> minEdge(N, INF);
        vector<char> inMST(N, 0);

        minEdge[startVertex] = 0.0;
        double totalWeight = 0.0;

        for (size_t step = 0; step < N; ++step) {
            size_t u = N;
            double minDist = INF;

            for (size_t v = 0; v < N; ++v) {
                if (!inMST[v] && minEdge[v] < minDist) {
                    minDist = minEdge[v];
                    u = v;
                }
            }

            if (u == N || minDist >= INF) {
                break;
            }

            inMST[u] = 1;
            totalWeight += minDist;

            for (size_t v = 0; v < N; ++v) {
                if (!inMST[v]) {
                    double weight = graph(u, v);
                    if (weight < minEdge[v]) {
                        minEdge[v] = weight;
                    }
                }
            }
        }

        return totalWeight;
    }
};

