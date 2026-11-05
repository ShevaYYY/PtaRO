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
#include "SequentialGaussSolver.hpp"
#include "ParallelGaussSolver.hpp"
#include "SequentialFloydSolver.hpp"
#include "ParallelFloydSolver.hpp"
#include "SequentialDijkstraSolver.hpp"
#include "ParallelDijkstraSolver.hpp"
#include "SequentialPrimSolver.hpp"
#include "ParallelPrimSolver.hpp"

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

void testGaussSolverCorrectnessAndEquivalence() {
    cout << "[Test 7] Sequential vs Parallel Gaussian Elimination correctness... ";
    
    // Known 3x3 SLAE:
    // 2x0 + 1x1 - 1x2 = 8
    // -3x0 - 1x1 + 2x2 = -11
    // -2x0 + 1x1 + 2x2 = -3
    // Exact solution: x0 = 2, x1 = 3, x2 = -1
    Matrix aug(3, 4);
    aug(0,0) = 2;  aug(0,1) = 1;  aug(0,2) = -1; aug(0,3) = 8;
    aug(1,0) = -3; aug(1,1) = -1; aug(1,2) = 2;  aug(1,3) = -11;
    aug(2,0) = -2; aug(2,1) = 1;  aug(2,2) = 2;  aug(2,3) = -3;

    SequentialGaussSolver seqSolver;
    ParallelGaussSolver parSolver(4);

    vector<double> xSeq = seqSolver.solve(aug);
    vector<double> xPar = parSolver.solve(aug);

    assert(xSeq.size() == 3);
    assert(xPar.size() == 3);

    assert(abs(xSeq[0] - 2.0) < 1e-9);
    assert(abs(xSeq[1] - 3.0) < 1e-9);
    assert(abs(xSeq[2] - (-1.0)) < 1e-9);

    assert(abs(xPar[0] - 2.0) < 1e-9);
    assert(abs(xPar[1] - 3.0) < 1e-9);
    assert(abs(xPar[2] - (-1.0)) < 1e-9);

    // Random larger well-conditioned matrix equivalence test (100 x 101)
    size_t N = 100;
    Matrix randAug = MatrixBenchmarkSuite::generateWellConditionedSLAE(N);

    vector<double> randSeqX = seqSolver.solve(randAug);
    vector<double> randParX = parSolver.solve(randAug);

    assert(randSeqX.size() == N);
    assert(randParX.size() == N);

    for (size_t i = 0; i < N; ++i) {
        assert(abs(randSeqX[i] - randParX[i]) < 1e-7);
    }

    // Zero pivot exception check
    Matrix singularAug(2, 3);
    singularAug(0, 0) = 0; singularAug(0, 1) = 0; singularAug(0, 2) = 5;
    singularAug(1, 0) = 0; singularAug(1, 1) = 0; singularAug(1, 2) = 10;

    bool caughtZeroPivot = false;
    try {
        seqSolver.solve(singularAug);
    } catch (const runtime_error&) {
        caughtZeroPivot = true;
    }
    assert(caughtZeroPivot);

    cout << "PASSED\n";
}

void testFloydWarshallCorrectnessAndEquivalence() {
    cout << "[Test 8] Sequential vs Parallel Floyd-Warshall correctness... ";

    // Known 4-vertex directed weighted graph
    // 0 -> 1 (5), 0 -> 3 (10), 1 -> 2 (3), 2 -> 3 (1)
    // Shortest path from 0 to 3 should be 0->1->2->3 = 9 (instead of direct 10)
    const double INF = SequentialFloydSolver::INF;
    Matrix graph(4, 4, INF);
    for (size_t i = 0; i < 4; ++i) {
        graph(i, i) = 0.0;
    }
    graph(0, 1) = 5.0;
    graph(0, 3) = 10.0;
    graph(1, 2) = 3.0;
    graph(2, 3) = 1.0;

    SequentialFloydSolver seqSolver;
    ParallelFloydSolver parSolver(4);

    Matrix distSeq = seqSolver.solve(graph);
    Matrix distPar = parSolver.solve(graph);

    assert(distSeq.getRows() == 4 && distSeq.getCols() == 4);
    assert(distPar.getRows() == 4 && distPar.getCols() == 4);

    // Verify known shortest distances
    assert(abs(distSeq(0, 0) - 0.0) < 1e-9);
    assert(abs(distSeq(0, 1) - 5.0) < 1e-9);
    assert(abs(distSeq(0, 2) - 8.0) < 1e-9);
    assert(abs(distSeq(0, 3) - 9.0) < 1e-9); // 0->1->2->3 is 9, not 10!

    assert(abs(distSeq(1, 0) - INF) < 1e-9);
    assert(abs(distSeq(1, 1) - 0.0) < 1e-9);
    assert(abs(distSeq(1, 2) - 3.0) < 1e-9);
    assert(abs(distSeq(1, 3) - 4.0) < 1e-9);

    assert(abs(distSeq(2, 0) - INF) < 1e-9);
    assert(abs(distSeq(2, 1) - INF) < 1e-9);
    assert(abs(distSeq(2, 2) - 0.0) < 1e-9);
    assert(abs(distSeq(2, 3) - 1.0) < 1e-9);

    assert(abs(distSeq(3, 0) - INF) < 1e-9);
    assert(abs(distSeq(3, 1) - INF) < 1e-9);
    assert(abs(distSeq(3, 2) - INF) < 1e-9);
    assert(abs(distSeq(3, 3) - 0.0) < 1e-9);

    // Verify parallel matches sequential on known graph
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(abs(distSeq(i, j) - distPar(i, j)) < 1e-9);
        }
    }

    // Equivalence test on a larger random graph
    size_t N = 80;
    Matrix randGraph = MatrixBenchmarkSuite::generateRandomGraph(N, 0.6);

    Matrix randSeqDist = seqSolver.solve(randGraph);
    Matrix randParDist = parSolver.solve(randGraph);

    assert(randSeqDist.getRows() == N && randSeqDist.getCols() == N);
    assert(randParDist.getRows() == N && randParDist.getCols() == N);

    for (size_t i = 0; i < randSeqDist.getTotalElements(); ++i) {
        assert(abs(randSeqDist.data()[i] - randParDist.data()[i]) < 1e-9);
    }

    // Invalid dimension exception check
    bool caughtException = false;
    try {
        Matrix nonSquare(3, 4);
        seqSolver.solve(nonSquare);
    } catch (const invalid_argument&) {
        caughtException = true;
    }
    assert(caughtException);

    cout << "PASSED\n";
}


void testDijkstraCorrectnessAndEquivalence() {
    cout << "[Test 9] Sequential vs Parallel Dijkstra correctness... ";

    // Known 5-vertex directed weighted graph (CLRS Example)
    // 0 -> 1 (10), 0 -> 4 (5), 1 -> 2 (1), 1 -> 4 (2), 4 -> 1 (3), 4 -> 2 (9), 4 -> 3 (2), 2 -> 3 (4), 3 -> 2 (6), 3 -> 0 (7)
    // Shortest paths from 0:
    // 0: 0, 1: 8 (0->4->1), 2: 9 (0->4->1->2), 3: 7 (0->4->3), 4: 5 (0->4)
    const double INF = SequentialDijkstraSolver::INF;
    Matrix graph(5, 5, INF);
    for (size_t i = 0; i < 5; ++i) {
        graph(i, i) = 0.0;
    }
    graph(0, 1) = 10.0;
    graph(0, 4) = 5.0;
    graph(1, 2) = 1.0;
    graph(1, 4) = 2.0;
    graph(4, 1) = 3.0;
    graph(4, 2) = 9.0;
    graph(4, 3) = 2.0;
    graph(2, 3) = 4.0;
    graph(3, 2) = 6.0;
    graph(3, 0) = 7.0;

    SequentialDijkstraSolver seqSolver;
    ParallelDijkstraSolver parSolver(4);

    vector<double> distSeq = seqSolver.solve(graph, 0);
    vector<double> distPar = parSolver.solve(graph, 0);

    assert(distSeq.size() == 5);
    assert(distPar.size() == 5);

    // Verify known shortest distances
    assert(abs(distSeq[0] - 0.0) < 1e-9);
    assert(abs(distSeq[1] - 8.0) < 1e-9);
    assert(abs(distSeq[2] - 9.0) < 1e-9);
    assert(abs(distSeq[3] - 7.0) < 1e-9);
    assert(abs(distSeq[4] - 5.0) < 1e-9);

    for (size_t i = 0; i < 5; ++i) {
        assert(abs(distSeq[i] - distPar[i]) < 1e-9);
    }

    // Verify unreachable node in disconnected graph
    Matrix discGraph(3, 3, INF);
    discGraph(0, 0) = 0.0; discGraph(1, 1) = 0.0; discGraph(2, 2) = 0.0;
    discGraph(0, 1) = 4.0; // node 2 is disconnected
    vector<double> discSeq = seqSolver.solve(discGraph, 0);
    vector<double> discPar = parSolver.solve(discGraph, 0);
    assert(abs(discSeq[0] - 0.0) < 1e-9);
    assert(abs(discSeq[1] - 4.0) < 1e-9);
    assert(abs(discSeq[2] - INF) < 1e-9);
    assert(abs(discPar[2] - INF) < 1e-9);

    // Equivalence test on a larger random graph (N = 100) with different thread counts
    size_t N = 100;
    Matrix randGraph = MatrixBenchmarkSuite::generateRandomGraph(N, 0.6);

    vector<double> randSeq = seqSolver.solve(randGraph, 0);

    for (size_t threads : {1, 2, 4, 8}) {
        ParallelDijkstraSolver testPar(threads);
        vector<double> randPar = testPar.solve(randGraph, 0);
        assert(randPar.size() == N);
        for (size_t i = 0; i < N; ++i) {
            assert(abs(randSeq[i] - randPar[i]) < 1e-7);
        }
    }

    // Exception handling checks
    bool caughtNonSquare = false;
    try {
        Matrix nonSquare(3, 4);
        seqSolver.solve(nonSquare, 0);
    } catch (const invalid_argument&) {
        caughtNonSquare = true;
    }
    assert(caughtNonSquare);

    bool caughtOutOfRange = false;
    try {
        seqSolver.solve(graph, 10);
    } catch (const out_of_range&) {
        caughtOutOfRange = true;
    }
    assert(caughtOutOfRange);

    cout << "PASSED\n";
}


void testPrimCorrectnessAndEquivalence() {
    cout << "[Test 10] Sequential vs Parallel Prim's MST correctness... ";

    // Known 5-vertex undirected graph (MST weight = 16)
    // Vertices: 0, 1, 2, 3, 4
    // Edges:
    // (0-1): 2, (0-3): 6
    // (1-2): 3, (1-3): 8, (1-4): 5
    // (2-4): 7
    // (3-4): 9
    // Expected MST edges: (0,1)=2, (1,2)=3, (1,4)=5, (0,3)=6. Total weight = 2+3+5+6 = 16.
    const double INF = SequentialPrimSolver::INF;
    Matrix graph(5, 5, INF);
    for (size_t i = 0; i < 5; ++i) {
        graph(i, i) = 0.0;
    }
    auto addEdge = [&](size_t u, size_t v, double w) {
        graph(u, v) = w;
        graph(v, u) = w;
    };
    addEdge(0, 1, 2.0);
    addEdge(0, 3, 6.0);
    addEdge(1, 2, 3.0);
    addEdge(1, 3, 8.0);
    addEdge(1, 4, 5.0);
    addEdge(2, 4, 7.0);
    addEdge(3, 4, 9.0);

    SequentialPrimSolver seqSolver;
    ParallelPrimSolver parSolver(4);

    double mstSeq = seqSolver.solve(graph, 0);
    double mstPar = parSolver.solve(graph, 0);

    assert(abs(mstSeq - 16.0) < 1e-9);
    assert(abs(mstPar - 16.0) < 1e-9);

    // Test starting from different root vertex (e.g. vertex 2)
    double mstSeqFrom2 = seqSolver.solve(graph, 2);
    double mstParFrom2 = parSolver.solve(graph, 2);
    assert(abs(mstSeqFrom2 - 16.0) < 1e-9);
    assert(abs(mstParFrom2 - 16.0) < 1e-9);

    // Disconnected graph test
    Matrix discGraph(4, 4, INF);
    for (size_t i = 0; i < 4; ++i) discGraph(i, i) = 0.0;
    discGraph(0, 1) = 3.0; discGraph(1, 0) = 3.0; // component 1: {0, 1}
    discGraph(2, 3) = 4.0; discGraph(3, 2) = 4.0; // component 2: {2, 3}
    double discSeq = seqSolver.solve(discGraph, 0);
    double discPar = parSolver.solve(discGraph, 0);
    assert(abs(discSeq - 3.0) < 1e-9);
    assert(abs(discPar - 3.0) < 1e-9);

    // Equivalence test on a larger random symmetric graph (N = 80) across thread counts
    size_t N = 80;
    Matrix randGraph = MatrixBenchmarkSuite::generateRandomSymmetricGraph(N, 0.6);

    double randSeq = seqSolver.solve(randGraph, 0);

    for (size_t threads : {1, 2, 4, 8}) {
        ParallelPrimSolver testPar(threads);
        double randPar = testPar.solve(randGraph, 0);
        assert(abs(randSeq - randPar) < 1e-7);
    }

    // Exception handling checks
    bool caughtNonSquare = false;
    try {
        Matrix nonSquare(3, 4);
        seqSolver.solve(nonSquare, 0);
    } catch (const invalid_argument&) {
        caughtNonSquare = true;
    }
    assert(caughtNonSquare);

    bool caughtOutOfRange = false;
    try {
        seqSolver.solve(graph, 10);
    } catch (const out_of_range&) {
        caughtOutOfRange = true;
    }
    assert(caughtOutOfRange);

    cout << "PASSED\n";
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
    testGaussSolverCorrectnessAndEquivalence();
    testFloydWarshallCorrectnessAndEquivalence();
    testDijkstraCorrectnessAndEquivalence();
    testPrimCorrectnessAndEquivalence();
    cout << "========================================\n";
    cout << " ALL VERIFICATION TESTS PASSED SUCCESSFULLY! \n";
    cout << "========================================\n";
    return 0;
}
