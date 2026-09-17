#pragma once

#include "IMatrixOperation.hpp"
#include <vector>
#include <future>
#include <stdexcept>
#include <iostream>

using namespace std;


class ChainMatrixAddOperation {
private:
    Matrix addRangeParallel(size_t start, size_t end, const vector<Matrix>& matrices, const IMatrixOperation& op) const {
        size_t count = end - start;
        if (count == 0) {
            throw invalid_argument("Empty chain range provided!");
        }
        if (count == 1) {
            return matrices[start];
        }
        if (count == 2) {
            return op.execute(matrices[start], matrices[start + 1]);
        }

        size_t mid = start + count / 2;

        auto leftFuture = async(launch::async, [this, start, mid, &matrices, &op]() {
            return addRangeParallel(start, mid, matrices, op);
        });

        Matrix rightResult = addRangeParallel(mid, end, matrices, op);
        Matrix leftResult = leftFuture.get();

        return op.execute(leftResult, rightResult);
    }

public:

    Matrix executeParallel(const vector<Matrix>& matrices, const IMatrixOperation& op) const {
        if (matrices.empty()) {
            throw invalid_argument("Matrix chain cannot be empty!");
        }
        return addRangeParallel(0, matrices.size(), matrices, op);
    }

    Matrix executeSequential(const vector<Matrix>& matrices, const IMatrixOperation& op) const {
        if (matrices.empty()) {
            throw invalid_argument("Matrix chain cannot be empty!");
        }

        Matrix result = matrices[0];
        for (size_t i = 1; i < matrices.size(); ++i) {
            result = op.execute(result, matrices[i]);
        }
        return result;
    }
};
