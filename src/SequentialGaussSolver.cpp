#include "../include/SequentialGaussSolver.hpp"

vector<double> SequentialGaussSolver::solve(const Matrix& augmented) const {
    size_t rows = augmented.getRows();
    size_t cols = augmented.getCols();

    if (rows == 0 || cols != rows + 1) {
        throw invalid_argument("Invalid augmented matrix dimensions for SLAE! Expected N x (N + 1).");
    }

    Matrix mat = augmented; // Working copy
    const size_t N = rows;
    const double EPS = 1e-12;

    // 1. Forward Elimination with Partial Pivoting
    for (size_t k = 0; k < N; ++k) {
        // Find pivot element in column k from row k to N-1
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

        // Swap current row k with pivotRow if needed
        if (pivotRow != k) {
            for (size_t j = k; j <= N; ++j) {
                swap(mat(k, j), mat(pivotRow, j));
            }
        }

        // Eliminate column k for lower rows i = k+1 .. N-1
        for (size_t i = k + 1; i < N; ++i) {
            double factor = mat(i, k) / mat(k, k);
            mat(i, k) = 0.0;
            for (size_t j = k + 1; j <= N; ++j) {
                mat(i, j) -= factor * mat(k, j);
            }
        }
    }

    // 2. Back Substitution
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
