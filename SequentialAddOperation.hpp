#pragma once

#include "IMatrixOperation.hpp"
#include <stdexcept>
#include <algorithm>

using namespace std;

class SequentialAddOperation : public IMatrixOperation {
public:
    Matrix execute(const Matrix& A, const Matrix& B) const override {
        if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
            throw invalid_argument("Matrix dimension mismatch for addition!");
        }

        Matrix result(A.getRows(), A.getCols());
        const auto& dataA = A.data();
        const auto& dataB = B.data();
        auto& dataRes = result.data();

        size_t total = A.getTotalElements();
        for (size_t i = 0; i < total; ++i) {
            dataRes[i] = dataA[i] + dataB[i];
        }

        return result;
    }

    string getName() const override {
        return "Sequential Add (Baseline)";
    }
};
