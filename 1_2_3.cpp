#include <iostream>
#include <vector>
#include <limits>
#include <chrono>

#include "Matrix.hpp"
#include "SequentialAddOperation.hpp"
#include "ParallelAddOperation.hpp"
#include "ChainMatrixAddOperation.hpp"
#include "MatrixBenchmarkSuite.hpp"

void printMenu() {
    std::cout << "\n=========================================================\n";
    std::cout << "    C++17 CONSOLE MATRIX ADDITION & BENCHMARK SUITE      \n";
    std::cout << "=========================================================\n";
    std::cout << " [1] Згенерувати матриці (Розмірність R x C)\n";
    std::cout << " [2] Лінійне додавання (Baseline, 1 потік, без async/thread)\n";
    std::cout << " [3] Паралельне додавання (Ввід кількості потоків вручну)\n";
    std::cout << " [4] Авто-бенчмарк (Адаптивні кроки від 1 до N потоків)\n";
    std::cout << " [5] Бенчмарк ланцюжка матриць (A + B + C + D... Бінарне дерево vs Лінійне)\n";
    std::cout << " [0] Вихід\n";
    std::cout << "=========================================================\n";
    std::cout << " Оберіть опцію: ";
}

void clearInputBuffer() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int main() {
    // Set UTF-8 encoding support if possible
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    Matrix matrixA;
    Matrix matrixB;
    bool matricesGenerated = false;

    int choice = -1;
    while (choice != 0) {
        printMenu();
        if (!(std::cin >> choice)) {
            std::cout << "[Помилка] Некоректний ввід. Спробуйте ще раз.\n";
            clearInputBuffer();
            continue;
        }

        try {
            switch (choice) {
                case 1: {
                    size_t rows = 0, cols = 0;
                    std::cout << "\n--- Генерація матриць A та B ---\n";
                    std::cout << "Введіть кількість рядків (R): ";
                    if (!(std::cin >> rows) || rows == 0) {
                        std::cout << "[Помилка] Рядки мають бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }
                    std::cout << "Введіть кількість стовпчиків (C): ";
                    if (!(std::cin >> cols) || cols == 0) {
                        std::cout << "[Помилка] Стовпчики мають бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }

                    matrixA = Matrix(rows, cols);
                    matrixB = Matrix(rows, cols);

                    matrixA.randomize(1.0, 100.0);
                    matrixB.randomize(1.0, 100.0);
                    matricesGenerated = true;

                    std::cout << "\n[Успіх] Згенеровано 2 матриці розмірністю " << rows << "x" << cols 
                              << " (" << matrixA.getTotalElements() << " елементів).\n";
                    
                    std::cout << "\nМатриця A (фрагмент):\n";
                    matrixA.print(4, 4);
                    std::cout << "Матриця B (фрагмент):\n";
                    matrixB.print(4, 4);
                    break;
                }
                case 2: {
                    if (!matricesGenerated) {
                        std::cout << "[Увага] Спочатку згенеруйте матриці (Опція 1).\n";
                        break;
                    }
                    std::cout << "\n--- Лінійне додавання (Baseline) ---\n";
                    SequentialAddOperation seqOp;

                    auto start = std::chrono::high_resolution_clock::now();
                    Matrix result = seqOp.execute(matrixA, matrixB);
                    auto end = std::chrono::high_resolution_clock::now();
                    double elapsedUs = std::chrono::duration<double, std::micro>(end - start).count();

                    std::cout << "Час виконання: " << std::fixed << std::setprecision(2) 
                              << elapsedUs << " us (" << (elapsedUs / 1000.0) << " ms)\n";
                    std::cout << "Результат (фрагмент):\n";
                    result.print(4, 4);
                    break;
                }
                case 3: {
                    if (!matricesGenerated) {
                        std::cout << "[Увага] Спочатку згенеруйте матриці (Опція 1).\n";
                        break;
                    }
                    size_t threads = 1;
                    std::cout << "\n--- Паралельне додавання (std::thread) ---\n";
                    std::cout << "Введіть кількість потоків: ";
                    if (!(std::cin >> threads) || threads == 0) {
                        std::cout << "[Помилка] Кількість потоків має бути > 0!\n";
                        clearInputBuffer();
                        break;
                    }

                    ParallelAddOperation parOp(threads);
                    auto start = std::chrono::high_resolution_clock::now();
                    Matrix result = parOp.execute(matrixA, matrixB);
                    auto end = std::chrono::high_resolution_clock::now();
                    double elapsedUs = std::chrono::duration<double, std::micro>(end - start).count();

                    std::cout << "Запитано потоків: " << threads << "\n";
                    std::cout << "Фактично використано: " << std::min(matrixA.getTotalElements(), threads) << "\n";
                    std::cout << "Час виконання: " << std::fixed << std::setprecision(2) 
                              << elapsedUs << " us (" << (elapsedUs / 1000.0) << " ms)\n";
                    std::cout << "Результат (фрагмент):\n";
                    result.print(4, 4);
                    break;
                }
                case 4: {
                    if (!matricesGenerated) {
                        std::cout << "[Увага] Спочатку згенеруйте матриці (Опція 1).\n";
                        break;
                    }
                    size_t maxThreads = 240;
                    std::cout << "\n--- Авто-бенчмарк (Smart Step Sampling) ---\n";
                    std::cout << "Введіть максимальну кількість потоків для бенчмарку (за замовчуванням 240): ";
                    clearInputBuffer();
                    std::string inputStr;
                    std::getline(std::cin, inputStr);
                    if (!inputStr.empty()) {
                        try {
                            maxThreads = std::stoull(inputStr);
                        } catch (...) {
                            maxThreads = 240;
                        }
                    }

                    MatrixBenchmarkSuite::runAutoBenchmark(matrixA, matrixB, maxThreads);
                    break;
                }
                case 5: {
                    std::cout << "\n--- Бенчмарк додавання ланцюжка матриць ---\n";
                    size_t count = 4, r = 1000, c = 1000;
                    std::cout << "Кількість матриць у ланцюжку (наприклад, 4, 8, 16): ";
                    if (!(std::cin >> count) || count < 2) {
                        count = 4;
                        std::cout << "Встановлено за замовчуванням: 4\n";
                    }
                    std::cout << "Розмірність матриць (R C, наприклад 1000 1000): ";
                    if (!(std::cin >> r >> c) || r == 0 || c == 0) {
                        r = 1000; c = 1000;
                        std::cout << "Встановлено за замовчуванням: 1000 x 1000\n";
                    }

                    std::cout << "Генерація " << count << " матриць " << r << "x" << c << "...\n";
                    std::vector<Matrix> chain;
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
                    std::cout << "\nЗавершення роботи програми. До побачення!\n";
                    break;
                default:
                    std::cout << "[Помилка] Невірний вибір опції.\n";
                    break;
            }
        } catch (const std::exception& ex) {
            std::cout << "\n[Exception]: " << ex.what() << "\n";
        }
    }

    return 0;
}
