#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#include "Matrix.hpp"
#include "ThreadChunkDistributor.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "SequentialMultiplyOperation.hpp"
#include "ParallelMultiplyOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "MatrixBenchmarkSuite.hpp"

using namespace std;

void testMatrixBasic() {
    cout << "[Test 1] Matrix basic allocation & indexing... ";
    Matrix m(3, 4, 2.5);
    assert(m.getRows() == 3);
    assert(m.getCols() == 4);
    assert(m.getTotalElements() == 12);
    assert(abs(m(1, 2) - 2.5) < 1e-9);

    m(1, 2) = 10.0;
    assert(abs(m(1, 2) - 10.0) < 1e-9);
    assert(abs(m.data()[1 * 4 + 2] - 10.0) < 1e-9);
    cout << "PASSED\n";
}

void testChunkDistributor() {
    cout << "[Test 2] ThreadChunkDistributor 1D partitioning... ";
    auto chunks = ThreadChunkDistributor::distribute(10, 3);
    assert(chunks.size() == 3);
    assert(chunks[0].start == 0 && chunks[0].end == 4);
    assert(chunks[1].start == 4 && chunks[1].end == 7);
    assert(chunks[2].start == 7 && chunks[2].end == 10);

    auto chunksEdge = ThreadChunkDistributor::distribute(2, 8);
    assert(chunksEdge.size() == 2);
    assert(chunksEdge[0].start == 0 && chunksEdge[0].end == 1);
    assert(chunksEdge[1].start == 1 && chunksEdge[1].end == 2);

    cout << "PASSED\n";
}

void testAdditionEquivalence() {
    cout << "[Test 3] Sequential vs Parallel addition equivalence... ";
    Matrix A(500, 500);
    Matrix B(500, 500);
    A.randomize(1.0, 100.0);
    B.randomize(1.0, 100.0);

    SequentialAddOperation seqOp;
    ParallelAddOperation parOp(8);

    Matrix resSeq = seqOp.execute(A, B);
    Matrix resPar = parOp.execute(A, B);

    assert(resSeq.getRows() == resPar.getRows());
    assert(resSeq.getCols() == resPar.getCols());

    for (size_t i = 0; i < resSeq.getTotalElements(); ++i) {
        assert(abs(resSeq.data()[i] - resPar.data()[i]) < 1e-9);
    }
    cout << "PASSED\n";
}

void testMultiplicationCorrectnessAndEquivalence() {
    cout << "[Test 4] Sequential vs Parallel matrix multiplication correctness... ";
    // Small known multiplication test:
    // A (2x3) * B (3x2) = C (2x2)
    Matrix A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

    Matrix B(3, 2);
    B(0, 0) = 7; B(0, 1) = 8;
    B(1, 0) = 9; B(1, 1) = 1;
    B(2, 0) = 2; B(2, 1) = 3;

    SequentialMultiplyOperation seqMultOp;
    Matrix C = seqMultOp.execute(A, B);

    // Expected C:
    // C(0,0) = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C(0,1) = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C(1,0) = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C(1,1) = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert(abs(C(0, 0) - 31.0) < 1e-9);
    assert(abs(C(0, 1) - 19.0) < 1e-9);
    assert(abs(C(1, 0) - 85.0) < 1e-9);
    assert(abs(C(1, 1) - 55.0) < 1e-9);

    // Random larger matrix equivalence test
    Matrix randA(150, 120);
    Matrix randB(120, 180);
    randA.randomize(1.0, 50.0);
    randB.randomize(1.0, 50.0);

    ParallelMultiplyOperation parMultOp(6);
    Matrix resSeq = seqMultOp.execute(randA, randB);
    Matrix resPar = parMultOp.execute(randA, randB);

    assert(resSeq.getRows() == resPar.getRows());
    assert(resSeq.getCols() == resPar.getCols());

    for (size_t i = 0; i < resSeq.getTotalElements(); ++i) {
        assert(abs(resSeq.data()[i] - resPar.data()[i]) < 1e-9);
    }

    // Dimension mismatch exception check
    bool caughtException = false;
    try {
        Matrix badB(100, 180); // rowsB=100 != colsA=120
        seqMultOp.execute(randA, badB);
    } catch (const invalid_argument&) {
        caughtException = true;
    }
    assert(caughtException);

    cout << "PASSED\n";
}

void testChainAddition() {
    cout << "[Test 5] Chain matrix binary tree addition equivalence... ";
    vector<Matrix> chain;
    for (int i = 0; i < 4; ++i) {
        Matrix m(200, 200);
        m.randomize(1.0, 10.0);
        chain.push_back(m);
    }

    ChainMatrixAddOperation chainOp;
    SequentialAddOperation seqSubOp;

    Matrix seqChainRes = chainOp.executeSequential(chain, seqSubOp);
    Matrix treeChainRes = chainOp.executeParallel(chain, seqSubOp);

    for (size_t i = 0; i < seqChainRes.getTotalElements(); ++i) {
        assert(abs(seqChainRes.data()[i] - treeChainRes.data()[i]) < 1e-9);
    }
    cout << "PASSED\n";
}

void testBenchmarkSampling() {
    cout << "[Test 6] Adaptive thread sampling generation... ";
    auto steps = MatrixBenchmarkSuite::generateAdaptiveThreadCounts(240);
    assert(!steps.empty());
    assert(steps.front() == 1);
    assert(steps.back() == 240);
    cout << "PASSED (Total samples up to 240 threads: " << steps.size() << ")\n";
}

int main() {
    cout << "========================================\n";
    cout << "  RUNNING AUTOMATED UNIT & INTEGRATION TESTS  \n";
    cout << "========================================\n";
    testMatrixBasic();
    testChunkDistributor();
    testAdditionEquivalence();
    testMultiplicationCorrectnessAndEquivalence();
    testChainAddition();
    testBenchmarkSampling();
    cout << "========================================\n";
    cout << " ALL VERIFICATION TESTS PASSED SUCCESSFULLY! \n";
    cout << "========================================\n";
    return 0;
}
