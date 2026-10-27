#pragma once

#include "Matrix.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>

using namespace std;

class SequentialFloydSolver {
public:
    static constexpr double INF = 1e9;

    SequentialFloydSolver() = default;

    Matrix solve(const Matrix& graph) const {
        size_t rows = graph.getRows();
        size_t cols = graph.getCols();

        if (rows == 0 || rows != cols) {
            throw invalid_argument("Invalid graph dimensions for Floyd-Warshall! Expected square matrix N x N with N > 0.");
        }

        Matrix dist = graph;
        const size_t N = rows;

        for (size_t k = 0; k < N; ++k) {
            for (size_t i = 0; i < N; ++i) {
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
        }

        return dist;
    }
};

