#include <iostream>
#include <opencv2/opencv.hpp>

#include "gray_scale.h"

#define OUTPUT_FILE "out.png"

/**
 * @brief Convert @c cv::Mat to a C-style 3D @c uchar arrays.
 *
 * @param mat
 * @return an array of three @c uchar** pointer, each of which represents a
 * pixel matrix.
 */
uchar*** MatTo3D(const cv::Mat &mat) {
    int rows = mat.rows;
    int cols = mat.cols;
    uchar*** res = new uchar**[rows];
    for (int i = 0; i < rows; ++i) {
        res[i] = new uchar*[cols];
        for (int j = 0; j < cols; ++j) {
            res[i][j] = new uchar[3];
            cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; ++k) {
                res[i][j][k] = pixel[k];
            }
        }
    }

    return res;
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " IMAGE" << std::endl;
        return -1;
    }
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    if (image.channels() != 3) {
        std::cerr << "The image is not 3-channel!" << std::endl;
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;

    std::cout << "Image " << argv[1] << ": " << rows << "x" << cols << std::endl;

    uchar*** image_bgr = MatTo3D(image);
    uchar** image_gray = new uchar*[rows];
    for (int i = 0; i < rows; ++i) {
        image_gray[i] = new uchar[cols];
    }

    BGRToGrayscale(image_gray, image_bgr, cols, rows);

    // Print the pixel value of the top-left 5x5 grids.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            uchar blue = image_bgr[i][j][0];
            uchar green = image_bgr[i][j][1];
            uchar red = image_bgr[i][j][2];
            uchar gray = image_gray[i][j];

            if (i < 5 && j < 5) {
                std::cout << "Pixel value at (" << i << ", " << j << "): ";
                std::cout << "B: " << static_cast<int>(blue) << ", ";
                std::cout << "G: " << static_cast<int>(green) << ", ";
                std::cout << "R: " << static_cast<int>(red) << ", ";
                std::cout << "Gray: " << static_cast<int>(gray) << std::endl;
            }
        }
    }

    cv::Mat img_res(rows, cols, CV_8UC1); // 8-bit, single channel
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            img_res.at<uchar>(i, j) = image_gray[i][j];
        }
    }
    cv::imwrite(OUTPUT_FILE, img_res);

    // FIXME: release memory of `image_bgr' and `image_gray'
    return 0;
}
