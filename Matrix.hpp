#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

using namespace std;

class Matrix {
private:
    size_t rows_;
    size_t cols_;
    vector<double> data_;

public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(size_t rows, size_t cols, double initVal = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, initVal) {}

    size_t getRows() const { return rows_; }
    size_t getCols() const { return cols_; }
    size_t getTotalElements() const { return data_.size(); }

    vector<double>& data() { return data_; }
    const vector<double>& data() const { return data_; }

    double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    double& at(size_t r, size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw out_of_range("Matrix index out of bounds (" + to_string(r) + ", " + to_string(c) + ")");
        }
        return data_[r * cols_ + c];
    }

    const double& at(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw out_of_range("Matrix index out of bounds (" + to_string(r) + ", " + to_string(c) + ")");
        }
        return data_[r * cols_ + c];
    }


    void randomize(double minVal = -10.0, double maxVal = 100.0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(minVal, maxVal);

        for (auto& val : data_) {
            val = dis(gen);
        }
    }


    void print(size_t maxRows = 5, size_t maxCols = 5) const {
        cout << "Matrix (" << rows_ << "x" << cols_ << "):\n";
        size_t displayRows = min(rows_, maxRows);
        size_t displayCols = min(cols_, maxCols);

        for (size_t r = 0; r < displayRows; ++r) {
            cout << "  [ ";
            for (size_t c = 0; c < displayCols; ++c) {
                cout << fixed << setprecision(2) << setw(8) << (*this)(r, c) << " ";
            }
            if (cols_ > maxCols) {
                cout << "... ";
            }
            cout << "]\n";
        }
        if (rows_ > maxRows) {
            cout << "  [ ... ]\n";
        }
        cout << endl;
    }
};

