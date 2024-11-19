# Matrix Multiplication

## Build

Make sure the googletest library is foundable. You may need to specify `CMAKE_PREFIX_PATH`.

``` shell
cmake -S . -B build
cmake --build build --target test_matmul
```

## Test

``` shell
./build/test_matmul
```
