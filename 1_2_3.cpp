#include <iostream>
#include <vector>
#include <limits>
#include <chrono>

#include "Matrix.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "MatrixBenchmarkSuite.hpp"

using namespace std;
using namespace std::chrono;

void printMenu() {
    cout << "\n=========================================================\n";
    cout << "    C++17 CONSOLE MATRIX ADDITION & BENCHMARK SUITE      \n";
    cout << "=========================================================\n";
    cout << " [1] Згенерувати матриці (Розмірність R x C)\n";
    cout << " [2] Лінійне додавання (Baseline, 1 потік, без async/thread)\n";
    cout << " [3] Паралельне додавання (Ввід кількості потоків вручну)\n";
    cout << " [4] Авто-бенчмарк (Адаптивні кроки від 1 до N потоків)\n";
    cout << " [5] Бенчмарк ланцюжка матриць (A + B + C + D... Бінарне дерево vs Лінійне)\n";
    cout << " [0] Вихід\n";
    cout << "=========================================================\n";
    cout << " Оберіть опцію: ";
}

void clearInputBuffer() {
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
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
                    size_t maxThreads = 240;
                    cout << "\n--- Авто-бенчмарк (Smart Step Sampling) ---\n";
                    cout << "Введіть максимальну кількість потоків для бенчмарку (за замовчуванням 240): ";
                    clearInputBuffer();
                    string inputStr;
                    getline(cin, inputStr);
                    if (!inputStr.empty()) {
                        try {
                            maxThreads = stoull(inputStr);
                        } catch (...) {
                            maxThreads = 240;
                        }
                    }

                    MatrixBenchmarkSuite::runAutoBenchmark(matrixA, matrixB, maxThreads);
                    break;
                }
                case 5: {
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
