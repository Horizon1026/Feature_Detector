#include "feature_harris.h"

namespace FEATURE_DETECTOR {

bool HarrisFeature::ComputeGradient(const Image &image) {
    Ix_.setZero(image.rows(), image.cols());
    Iy_.setZero(image.rows(), image.cols());

    for (int32_t row = options_.kHalfPatchSize; row < image.rows() - options_.kHalfPatchSize; ++row) {
        for (int32_t col = options_.kHalfPatchSize; col < image.cols() - options_.kHalfPatchSize; ++col) {
            Ix_(row, col) = image.GetPixelValueNoCheck<float>(row, col + 1) - image.GetPixelValueNoCheck<float>(row, col - 1);
            Iy_(row, col) = image.GetPixelValueNoCheck<float>(row + 1, col) - image.GetPixelValueNoCheck<float>(row - 1, col);
        }
    }

    return true;
}

float HarrisFeature::ComputeResponse(const Image &image,
                                     const int32_t row,
                                     const int32_t col) {

    Mat2 M = Mat2::Zero();
    int32_t cnt = 0;
    for (int32_t drow = - options_.kHalfPatchSize; drow <= options_.kHalfPatchSize; ++drow) {
        for (int32_t dcol = - options_.kHalfPatchSize; dcol <= options_.kHalfPatchSize; ++dcol) {
            const int32_t sub_row = row + drow;
            const int32_t sub_col = col + dcol;
            M(0, 0) += Ix_(sub_row, sub_col) * Ix_(sub_row, sub_col);
            M(0, 1) += Ix_(sub_row, sub_col) * Iy_(sub_row, sub_col);
            M(1, 1) += Iy_(sub_row, sub_col) * Iy_(sub_row, sub_col);
            ++cnt;
        }

    }

    M(1, 0) = M(0, 1);
    M /= cnt;

    const float det = M.determinant();
    const float trace = M.trace();
    return det - options_.k * trace * trace;
}

}