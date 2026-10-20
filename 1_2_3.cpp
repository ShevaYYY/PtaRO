#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <string>

#include "Matrix.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "SequentialMultiplyOperation.hpp"
#include "ParallelMultiplyOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "MatrixBenchmarkSuite.hpp"
#include "SequentialGaussSolver.hpp"
#include "ParallelGaussSolver.hpp"

using namespace std;
using namespace std::chrono;

void printMenu() {
    cout << "\n=========================================================\n";
    cout << "    C++17 CONSOLE MATRIX ADDITION & BENCHMARK SUITE      \n";
    cout << "=========================================================\n";
    cout << " [1] Згенерувати матриці A та B (R x C)\n";
    cout << " [2] Лінійне додавання (Baseline)\n";
    cout << " [3] Паралельне додавання (thread)\n";
    cout << " [4] Авто-бенчмарк ДОДАВАННЯ (Smart Step Sampling)\n";
    cout << " [5] Авто-бенчмарк МНОЖЕННЯ (Smart Step Sampling)\n";
    cout << " [6] Авто-бенчмарк СЛАР (метод Гаусса)\n";
    cout << " [7] Бенчмарк ланцюжка матриць\n";
    cout << " [0] Вихід\n";
    cout << "=========================================================\n";
    cout << " Оберіть опцію: ";
}

void clearInputBuffer() {
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

size_t readMaxThreads(size_t defaultVal = 240) {
    size_t val = defaultVal;
    if (cin >> val && val > 0) {
        return val;
    }
    clearInputBuffer();
    return defaultVal;
}

int main() {
    #ifdef _WIN32
        system("chcp 65001 > nul");
    #endif

    Matrix matrixA;
    Matrix matrixB;
    bool matricesGenerated = false;

    int choice = -1;
    while (choice != 0) {
        printMenu();
        if (!(cin >> choice)) {
            cout << "[Помилка] Некоректний ввід. Спробуйте ще раз.\n";
            clearInputBuffer();
            continue;
        }

        try {
            switch (choice) {
                case 1: {
                    size_t rows = 0, cols = 0;
                    cout << "\n--- Генерація матриць A та B ---\n";
                    cout << "Введіть кількість рядків (R): ";
                    if (!(cin >> rows) || rows == 0) {
                        cout << "[Помилка] Рядки мають бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }
                    cout << "Введіть кількість стовпчиків (C): ";
                    if (!(cin >> cols) || cols == 0) {
                        cout << "[Помилка] Стовпчики мають бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }

                    matrixA = Matrix(rows, cols);
                    matrixB = Matrix(rows, cols);

                    matrixA.randomize(1.0, 100.0);
                    matrixB.randomize(1.0, 100.0);
                    matricesGenerated = true;

                    cout << "\n[Успіх] Згенеровано 2 матриці розмірністю " << rows << "x" << cols 
                         << " (" << matrixA.getTotalElements() << " елементів).\n";
                    
                    cout << "\nМатриця A (фрагмент):\n";
                    matrixA.print(4, 4);
                    cout << "Матриця B (фрагмент):\n";
                    matrixB.print(4, 4);
                    break;
                }
                case 2: {
                    if (!matricesGenerated) {
                        cout << "[Увага] Спочатку згенеруйте матриці (Опція 1).\n";
                        break;
                    }
                    cout << "\n--- Лінійне додавання (Baseline) ---\n";
                    SequentialAddOperation seqOp;

                    auto start = high_resolution_clock::now();
                    Matrix result = seqOp.execute(matrixA, matrixB);
                    auto end = high_resolution_clock::now();
                    double elapsedUs = duration<double, micro>(end - start).count();

                    cout << "Час виконання: " << fixed << setprecision(2) 
                         << elapsedUs << " us (" << (elapsedUs / 1000.0) << " ms)\n";
                    cout << "Результат (фрагмент):\n";
                    result.print(4, 4);
                    break;
                }
                case 3: {
                    if (!matricesGenerated) {
                        cout << "[Увага] Спочатку згенеруйте матриці (Опція 1).\n";
                        break;
                    }
                    size_t threads = 1;
                    cout << "\n--- Паралельне додавання (thread) ---\n";
                    cout << "Введіть кількість потоків: ";
                    if (!(cin >> threads) || threads == 0) {
                        cout << "[Помилка] Кількість потоків має бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }

                    ParallelAddOperation parOp(threads);
                    auto start = high_resolution_clock::now();
                    Matrix result = parOp.execute(matrixA, matrixB);
                    auto end = high_resolution_clock::now();
                    double elapsedUs = duration<double, micro>(end - start).count();

                    cout << "Запитано потоків: " << threads << "\n";
                    cout << "Фактично використано: " << min(matrixA.getTotalElements(), threads) << "\n";
                    cout << "Час виконання: " << fixed << setprecision(2) 
                         << elapsedUs << " us (" << (elapsedUs / 1000.0) << " ms)\n";
                    cout << "Результат (фрагмент):\n";
                    result.print(4, 4);
                    break;
                }
                case 4: {
                    if (!matricesGenerated) {
                        cout << "[Увага] Спочатку згенеруйте матриці (Опція 1).\n";
                        break;
                    }
                    cout << "\n--- Авто-бенчмарк ДОДАВАННЯ (Smart Step Sampling) ---\n";
                    cout << "Введіть максимальну кількість потоків для бенчмарку (наприклад, 32 або 240): ";
                    size_t maxThreads = readMaxThreads(240);

                    MatrixBenchmarkSuite::runAutoBenchmark(matrixA, matrixB, maxThreads);
                    break;
                }
                case 5: {
                    cout << "\n--- Авто-бенчмарк МНОЖЕННЯ МАТРИЦЬ (Smart Step Sampling) ---\n";
                    bool useExisting = false;
                    if (matricesGenerated && matrixA.getCols() == matrixB.getRows()) {
                        cout << "Знайдено згенеровані матриці A (" << matrixA.getRows() << "x" << matrixA.getCols() 
                             << ") та B (" << matrixB.getRows() << "x" << matrixB.getCols() << ").\n";
                        cout << "Використати існуючі матриці A та B? (1 - Так, 0 - Ні, згенерувати нові): ";
                        int choiceExisting = 1;
                        if (cin >> choiceExisting && choiceExisting == 1) {
                            useExisting = true;
                        }
                    }

                    Matrix multA, multB;
                    if (useExisting) {
                        multA = matrixA;
                        multB = matrixB;
                    } else {
                        size_t rowsA = 0, colsA = 0, colsB = 0;
                        cout << "Введіть розмірність матриці A (Рядки N, Стовпчики M): ";
                        if (!(cin >> rowsA >> colsA) || rowsA == 0 || colsA == 0) {
                            cout << "[Помилка] Розмірності мають бути > 0!\n";
                            clearInputBuffer();
                            break;
                        }
                        cout << "Введіть кількість стовпчиків матриці B (K) [Рядки M = " << colsA << "]: ";
                        if (!(cin >> colsB) || colsB == 0) {
                            cout << "[Помилка] Стовпчики K мають бути > 0!\n";
                            clearInputBuffer();
                            break;
                        }

                        cout << "Генерація сумісних матриць A (" << rowsA << "x" << colsA << ") та B (" << colsA << "x" << colsB << ")...\n";
                        multA = Matrix(rowsA, colsA);
                        multB = Matrix(colsA, colsB);
                        multA.randomize(1.0, 50.0);
                        multB.randomize(1.0, 50.0);
                    }

                    cout << "Введіть максимальну кількість потоків для бенчмарку (наприклад, 32 або 240): ";
                    size_t maxThreads = readMaxThreads(240);

                    MatrixBenchmarkSuite::runMultiplyBenchmark(multA, multB, maxThreads);
                    break;
                }
                case 6: {
                    cout << "\n--- Авто-бенчмарк СЛАР (метод Гаусса) ---\n";
                    size_t systemSize = 500;
                    cout << "Введіть розмірність системи N (кількість рівнянь/змінних): ";
                    if (!(cin >> systemSize) || systemSize == 0) {
                        cout << "[Помилка] Розмірність N має бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }
                    cout << "Введіть максимальну кількість потоків для бенчмарку (наприклад, 32 або 240): ";
                    size_t maxThreads = readMaxThreads(240);

                    MatrixBenchmarkSuite::runGaussBenchmark(systemSize, maxThreads);
                    break;
                }
                case 7: {
                    cout << "\n--- Бенчмарк додавання ланцюжка матриць ---\n";
                    size_t count = 4, r = 1000, c = 1000;
                    cout << "Кількість матриць у ланцюжку (наприклад, 4, 8, 16): ";
                    if (!(cin >> count) || count < 2) {
                        count = 4;
                        cout << "Встановлено за замовчуванням: 4\n";
                    }
                    cout << "Розмірність матриць (R C, наприклад 1000 1000): ";
                    if (!(cin >> r >> c) || r == 0 || c == 0) {
                        r = 1000; c = 1000;
                        cout << "Встановлено за замовчуванням: 1000 x 1000\n";
                    }

                    cout << "Генерація " << count << " матриць " << r << "x" << c << "...\n";
                    vector<Matrix> chain;
                    chain.reserve(count);
                    for (size_t i = 0; i < count; ++i) {
                        Matrix m(r, c);
                        m.randomize(1.0, 50.0);
                        chain.push_back(m);
                    }

                    MatrixBenchmarkSuite::runChainBenchmark(chain);
                    break;
                }
                case 0:
                    cout << "\nЗавершення роботи програми. До побачення!\n";
                    break;
                default:
                    cout << "[Помилка] Невірний вибір опції.\n";
                    break;
            }
        } catch (const exception& ex) {
            cout << "\n[Exception]: " << ex.what() << "\n";
        }
    }

    return 0;
}
