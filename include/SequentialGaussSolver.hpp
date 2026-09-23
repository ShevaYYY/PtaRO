#pragma once

#include "../Matrix.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace std;

class SequentialGaussSolver {
public:
    SequentialGaussSolver() = default;

    /**
     * @brief Solves SLAE [A | b] using sequential Gaussian elimination method.
     * @param augmented Matrix of size N x (N + 1) representing [A | b].
     * @return std::vector<double> Solution vector x of size N.
     * @throws std::invalid_argument if dimensions are invalid (cols != rows + 1).
     * @throws std::runtime_error if zero pivot element is encountered.
     */
    vector<double> solve(const Matrix& augmented) const;
};
