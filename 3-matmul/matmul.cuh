#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <iostream>
#include <stdexcept>
#include <string>

template<typename T>
__global__ void MatMulKernel(T* A, T* B, T* C, int m, int n, int k) {
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int col = blockIdx.y*blockDim.y+threadIdx.y;

    // Calculate the value of (row, col) in C.
    if ((row < m) && (col < k)) {
        T v = 0.0;
        for (int i = 0; i < n; ++i) {
            v += A[row*n+i]*B[i*k+col];
        }
        C[row*k+col] = v;
    }
}

template<typename T>
void flattenMatrix(T* out, T** in, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i*n+j] = in[i][j];
        }
    }
}

/**
 * @brief Calculate AxB and store the result in C.
 *
 * @param C the result matrix, should be m x k
 * @param A the left matrix, should be m x n
 * @param B the right matrix, should be n x k
 */
template<typename T>
void MatMul(T** C, T** A, T** B, int m, int n, int k) {
    T *A_h, *B_h, *C_h, *A_d, *B_d, *C_d;
    A_h = new T[m*n];
    B_h = new T[n*k];
    C_h = new T[m*k];

    flattenMatrix<T>(A_h, A, m, n);
    flattenMatrix<T>(B_h, B, n, k);

    cudaMalloc((void**)&A_d, m*n*sizeof(T));
    cudaMalloc((void**)&B_d, n*k*sizeof(T));
    cudaMalloc((void**)&C_d, m*k*sizeof(T));

    cudaMemcpy(A_d, A_h, m*n*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n*k*sizeof(T), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(m/16.0), ceil(k/16.0), 1);
    MatMulKernel<T><<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m, n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string str{};
        str += "Kernel launch error: ";
        str += cudaGetErrorString(err);
        throw std::runtime_error{str};
    }

    cudaMemcpy(C_h, C_d, m*k*sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            C[i][j] = C_h[i*k+j];
        }
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    delete []A_h;
    delete []B_h;
    delete []C_h;
}

#endif
