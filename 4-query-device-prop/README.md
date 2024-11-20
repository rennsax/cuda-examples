# Query CUDA Device Properties

The small program will print the number of available CUDA devices and the properties of each device.

## Build

``` shell
nvcc main.cu -o main
```

## Result

For NVIDIA GeForce RTX 4090:

``` text
CUDA device count: 1

Device 0:
Max thread per block: 1024
SM count: 128
Clock frequency: 2520000
Registers per SM: 65536
Maximal block size: (1024, 1024, 64)
Maximal grid size: (2147483647, 65535, 65535)
Warp size: 32
```
