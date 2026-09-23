#pragma once

#include "../Matrix.hpp"
#include "../ThreadChunkDistributor.hpp"
#include <vector>
#include <thread>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace std;

class ParallelGaussSolver {
private:
    size_t requestedThreads_;

public:
    explicit ParallelGaussSolver(size_t requestedThreads = thread::hardware_concurrency())
        : requestedThreads_(requestedThreads > 0 ? requestedThreads : 1) {}

    void setRequestedThreads(size_t threads) {
        requestedThreads_ = threads > 0 ? threads : 1;
    }

    size_t getRequestedThreads() const {
        return requestedThreads_;
    }

    /**
     * @brief Solves SLAE [A | b] using parallel Gaussian elimination method.
     * Forward elimination is parallelized per-row at each elimination step k.
     * Back substitution is executed sequentially.
     * 
     * @param augmented Matrix of size N x (N + 1) representing [A | b].
     * @return std::vector<double> Solution vector x of size N.
     * @throws std::invalid_argument if dimensions are invalid (cols != rows + 1).
     * @throws std::runtime_error if zero pivot element is encountered.
     */
    vector<double> solve(const Matrix& augmented) const;
};
