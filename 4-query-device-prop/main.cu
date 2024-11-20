#include <iostream>

std::ostream& operator<<(std::ostream &os, const cudaDeviceProp &prop) {
    os << "Max thread per block: " << prop.maxThreadsPerBlock << '\n'
       << "SM count: " << prop.multiProcessorCount << '\n'
       << "Clock frequency: " << prop.clockRate << '\n'
       << "Registers per SM: " << prop.regsPerBlock << '\n'
       << "Maximal block size: (" << prop.maxThreadsDim[0] << ", "
       << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n"
       << "Maximal grid size: (" << prop.maxGridSize[0] << ", "
       << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n"
       << "Warp size: " << prop.warpSize << '\n';
    return os;
}

int main() {
    int devCnt;
    cudaGetDeviceCount(&devCnt);

    std::cout << "CUDA device count: " << devCnt << std::endl;

    for (unsigned i = 0; i < devCnt; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << '\n' << "Device " << i << ":\n" << devProp;
    }
    endl(std::cout);
}
