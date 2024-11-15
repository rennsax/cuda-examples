#ifndef GRAY_SCALE_H
#define GRAY_SCALE_H

using uchar = unsigned char;

constexpr unsigned int CHANNEL_NUMBER = 3;

/**
 * @brief Convert the image from BGR to grayscale.
 *
 * @param image the 3D array represent the image. It will be modified in-place.
 * @param width
 * @param height
 * @return out the grayscale image, single changle.
 */
void BGRToGrayscale(uchar** out, uchar*** image, int width, int height);

#endif
