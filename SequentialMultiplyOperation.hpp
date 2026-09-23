#pragma once

#include "IMatrixOperation.hpp"
#include <stdexcept>

using namespace std;

class SequentialMultiplyOperation : public IMatrixOperation {
public:
    Matrix execute(const Matrix& A, const Matrix& B) const override {
        if (A.getCols() != B.getRows()) {
            throw invalid_argument("Matrix dimension mismatch for multiplication! (colsA != rowsB)");
        }

        size_t rowsA = A.getRows();
        size_t colsA = A.getCols();
        size_t colsB = B.getCols();

        Matrix result(rowsA, colsB, 0.0);

        const auto& dataA = A.data();
        const auto& dataB = B.data();
        auto& dataRes = result.data();

        for (size_t i = 0; i < rowsA; ++i) {
            size_t rowOffsetA = i * colsA;
            size_t rowOffsetRes = i * colsB;
            for (size_t k = 0; k < colsA; ++k) {
                double valA = dataA[rowOffsetA + k];
                size_t rowOffsetB = k * colsB;
                for (size_t j = 0; j < colsB; ++j) {
                    dataRes[rowOffsetRes + j] += valA * dataB[rowOffsetB + j];
                }
            }
        }
        return result;
    }

    string getName() const override {
        return "Sequential Multiply (Baseline, i-k-j)";
    }
};
