# Multiple Dimension Grids: Grayscale Convertor

Convert colored images to garyscale.

## Build

``` shell
cmake -S . -B build
cmake --build build color2gray
```

## Test

The image for testing is available [on the Wikipedia](https://en.wikipedia.org/wiki/File:Lenna_(test_image).png).

``` shell
./build/color2gray ./Lenna.jpg
```

The output image is `out.png` at the same directory.
