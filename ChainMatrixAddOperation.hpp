#ifndef CHAIN_MATRIX_ADD_OPERATION_HPP
#define CHAIN_MATRIX_ADD_OPERATION_HPP

#include "IMatrixOperation.hpp"
#include <vector>
#include <future>
#include <stdexcept>
#include <iostream>

/**
 * @brief Summation of matrix chain A + B + C + D + ... using parallel binary tree task parallelism (std::async).
 */
class ChainMatrixAddOperation {
private:
    Matrix addRangeParallel(size_t start, size_t end, const std::vector<Matrix>& matrices, const IMatrixOperation& op) const {
        size_t count = end - start;
        if (count == 0) {
            throw std::invalid_argument("Empty chain range provided!");
        }
        if (count == 1) {
            return matrices[start];
        }
        if (count == 2) {
            return op.execute(matrices[start], matrices[start + 1]);
        }

        size_t mid = start + count / 2;

        // Launch left sub-tree asynchronously using task parallelism
        auto leftFuture = std::async(std::launch::async, [this, start, mid, &matrices, &op]() {
            return addRangeParallel(start, mid, matrices, op);
        });

        // Compute right sub-tree in the current task context
        Matrix rightResult = addRangeParallel(mid, end, matrices, op);
        Matrix leftResult = leftFuture.get();

        return op.execute(leftResult, rightResult);
    }

public:
    /**
     * @brief Computes sum of matrix chain via parallel binary tree (std::async).
     */
    Matrix executeParallel(const std::vector<Matrix>& matrices, const IMatrixOperation& op) const {
        if (matrices.empty()) {
            throw std::invalid_argument("Matrix chain cannot be empty!");
        }
        return addRangeParallel(0, matrices.size(), matrices, op);
    }

    /**
     * @brief Computes sum of matrix chain sequentially (linear fold baseline).
     */
    Matrix executeSequential(const std::vector<Matrix>& matrices, const IMatrixOperation& op) const {
        if (matrices.empty()) {
            throw std::invalid_argument("Matrix chain cannot be empty!");
        }

        Matrix result = matrices[0];
        for (size_t i = 1; i < matrices.size(); ++i) {
            result = op.execute(result, matrices[i]);
        }
        return result;
    }
};

#endif // CHAIN_MATRIX_ADD_OPERATION_HPP
