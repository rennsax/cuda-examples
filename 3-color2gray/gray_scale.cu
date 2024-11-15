#include <iostream>
#include <cassert>
#include <stdexcept>
#include <string>

#include "gray_scale.h"

__global__ void BGRToGrayscaleKernel(uchar* Pout, uchar* Pin, int width, int height) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int p = row*width + col; // (flattened) position of the pixel
        int i = p*3;
        uchar b = Pin[i];
        uchar g = Pin[i+1];
        uchar r = Pin[i+2];
        Pout[p] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

static void linearize3D(uchar* out, uchar*** arr, int x_dim, int y_dim, int z_dim) {

    int index = 0;
    for (int x = 0; x < x_dim; ++x) {
        for (int y = 0; y < y_dim; ++y) {
            for (int z = 0; z < z_dim; ++z) {
                out[index++] = arr[x][y][z];
            }
        }
    }
}

static void reshape2D(uchar** out, uchar* vec, int x_dim, int y_dim) {
    for (int x = 0; x < x_dim; ++x) {
        for (int y = 0; y < y_dim; ++y) {
            int i = (x*x_dim)+y;
            out[x][y] = vec[i];
        }
    }
}

void BGRToGrayscale(uchar** out, uchar*** image, int width, int height) {
    int n_pixel = width * height;
    int input_size = n_pixel * CHANNEL_NUMBER * sizeof(uchar);
    int output_size = n_pixel * sizeof(uchar);

    // This is used both for host-side input and output.
    assert(input_size >= output_size);
    uchar* P_h = new uchar[input_size];
    linearize3D(P_h, image, height, width, CHANNEL_NUMBER);

    uchar* Pin_d, *Pout_d;
    cudaMalloc((void **)&Pin_d, input_size);
    cudaMalloc((void **)&Pout_d, output_size);

    cudaMemcpy(Pin_d, P_h, input_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(height/16.0), ceil(width/16.0), 1);
    BGRToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::string str{};
        str += "Kernel launch error: ";
        str += cudaGetErrorString(err);
        throw std::runtime_error{str};
    }

    cudaMemcpy(P_h, Pout_d, output_size, cudaMemcpyDeviceToHost);
    reshape2D(out, P_h, height, width);

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    delete []P_h;
}
