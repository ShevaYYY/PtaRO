#pragma once

#include "Matrix.hpp"
#include <string>

using namespace std;


class IMatrixOperation {
public:
    virtual ~IMatrixOperation() = default;

    virtual Matrix execute(const Matrix& A, const Matrix& B) const = 0;

    virtual string getName() const = 0;
};
