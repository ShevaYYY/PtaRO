#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#include "Matrix.hpp"
#include "ThreadChunkDistributor.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "MatrixBenchmarkSuite.hpp"

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
    // 10 elements, 3 threads -> chunks of size 4, 3, 3
    assert(chunks.size() == 3);
    assert(chunks[0].start == 0 && chunks[0].end == 4);
    assert(chunks[1].start == 4 && chunks[1].end == 7);
    assert(chunks[2].start == 7 && chunks[2].end == 10);

    // Edge case: actualThreads = min(totalElements, requestedThreads)
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

void testChainAddition() {
    cout << "[Test 4] Chain matrix binary tree addition equivalence... ";
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
    cout << "[Test 5] Adaptive thread sampling generation... ";
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
    testChainAddition();
    testBenchmarkSampling();
    cout << "========================================\n";
    cout << " ALL VERIFICATION TESTS PASSED SUCCESSFULLY! \n";
    cout << "========================================\n";
    return 0;
}
