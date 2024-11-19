#include <gtest/gtest.h>
#include <cmath>

#include "random.hpp"

using Random = effolkronium::random_static;

template<typename T>
void MatMul(T** C, T** A, T** B, int m, int n, int k);

float** getRandomMatrix(int m, int n) {
    float** out = new float*[m];
    for (int i = 0; i < m; ++i) {
        out[i] = new float[n];
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i][j] = Random::get<Random::common>(0.0, 100.0f);
        }
    }

    return out;
}

void printMatrix(float** matrix, int m, int n) {
    // std::cout << std::fixed << std::setprecision(2) << std::setw(8);
    // std::cout << std::left << std::fixed << std::setprecision(2) << std::setw(8);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::left << std::fixed << std::setprecision(2) << std::setw(8) << matrix[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

void freeMatrix(float** p, int m) {
    for (int i = 0; i < m; ++i) {
        delete []p[i];
    }
    delete []p;
}

void matmul_cpu(float** C, float** A, float** B, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

TEST(TestMatrixMultiplication, Base) {
    int m = 5;
    int n = 6;
    float **A = getRandomMatrix(m, n), **B = getRandomMatrix(n, m);

    std::cout << "Matrix A:" << std::endl;
    printMatrix(A, m, n);
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    printMatrix(B, n, m);
    std::cout << std::endl;

    // Allocate space for the result matrix.
    float **C_res = new float*[m], **C_exp = new float*[m];
    for (int i = 0; i < m; ++i) {
        C_res[i] = new float[m];
        C_exp[i] = new float[m];
    }

    MatMul(C_res, A, B, m, n, m);
    std::cout << "Matrix C_res:" << std::endl;
    printMatrix(C_res, m, m);
    std::cout << std::endl;

    matmul_cpu(C_exp, A, B, m, n, m);
    std::cout << "Matrix C_exp:" << std::endl;
    printMatrix(C_exp, m, m);
    std::cout << std::endl;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            ASSERT_FLOAT_EQ(C_res[i][j], C_exp[i][j]);
        }
    }

    freeMatrix(A, m);
    freeMatrix(B, n);
    freeMatrix(C_res, m);
    freeMatrix(C_exp, m);
}
