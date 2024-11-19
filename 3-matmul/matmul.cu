#include "matmul.cuh"

template void MatMul<float>(float** C, float** A, float** B, int m, int n, int k);
