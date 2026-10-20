#pragma once

#include "Matrix.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

class SequentialGaussSolver {
public:
    SequentialGaussSolver() = default;

    vector<double> solve(const Matrix& augmented) const {
        size_t rows = augmented.getRows();
        size_t cols = augmented.getCols();
        if (rows == 0 || cols != rows + 1) {
            throw invalid_argument("Invalid augmented matrix dimensions for SLAE! Expected N x (N + 1).");
        }
        Matrix mat = augmented; 
        const size_t N = rows;
        const double EPS = 1e-12;

        for (size_t k = 0; k < N; ++k) {
            size_t pivotRow = k;
            double maxVal = abs(mat(k, k));
            for (size_t i = k + 1; i < N; ++i) {
                double val = abs(mat(i, k));
                if (val > maxVal) {
                    maxVal = val;
                    pivotRow = i;
                }
            }
            if (maxVal < EPS) {
                throw runtime_error("Zero pivot element encountered at step " + to_string(k) + 
                                    ": SLAE is singular or has no unique solution.");
            }
            if (pivotRow != k) {
                for (size_t j = k; j <= N; ++j) {
                    swap(mat(k, j), mat(pivotRow, j));
                }
            }
            for (size_t i = k + 1; i < N; ++i) {
                double factor = mat(i, k) / mat(k, k);
                mat(i, k) = 0.0;
                for (size_t j = k + 1; j <= N; ++j) {
                    mat(i, j) -= factor * mat(k, j);
                }
            }
        }
        vector<double> x(N, 0.0);
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            double sum = mat(i, N);
            for (size_t j = i + 1; j < N; ++j) {
                sum -= mat(i, j) * x[j];
            }
            x[i] = sum / mat(i, i);
        }
        return x;
    }
};
