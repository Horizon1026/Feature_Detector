#include "feature_point_shi_tomas_detector.h"

namespace feature_detector {

bool FeaturePointShiTomasDetector::ComputeGradient(const GrayImage &image) {
    RETURN_FALSE_IF(image.data() == nullptr);
    Ix_.setZero(image.rows(), image.cols());
    Iy_.setZero(image.rows(), image.cols());

    for (int32_t row = feature_options_.kHalfPatchSize; row < image.rows() - feature_options_.kHalfPatchSize; ++row) {
        for (int32_t col = feature_options_.kHalfPatchSize; col < image.cols() - feature_options_.kHalfPatchSize; ++col) {
            Ix_(row, col) = image.GetPixelValueNoCheck<float>(row, col + 1) - image.GetPixelValueNoCheck<float>(row, col - 1);
            Iy_(row, col) = image.GetPixelValueNoCheck<float>(row + 1, col) - image.GetPixelValueNoCheck<float>(row - 1, col);
        }
    }

    return true;
}

float FeaturePointShiTomasDetector::ComputeResponseOfPixel(const GrayImage &image, const int32_t row, const int32_t col) {
    Mat2 M = Mat2::Zero();
    int32_t cnt = 0;
    for (int32_t drow = -feature_options_.kHalfPatchSize; drow <= feature_options_.kHalfPatchSize; ++drow) {
        for (int32_t dcol = -feature_options_.kHalfPatchSize; dcol <= feature_options_.kHalfPatchSize; ++dcol) {
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

    Eigen::SelfAdjointEigenSolver<Mat2> saes(M);
    const Vec2 eig = saes.eigenvalues();
    return std::max(eig(0), eig(1));
}

bool FeaturePointShiTomasDetector::ComputeCandidates(const GrayImage &image) {
    if (ComputeGradient(image) == false) {
        return false;
    }

    const int32_t bound = feature_options_.kHalfPatchSize;
    for (int32_t row = bound; row < image.rows() - bound; ++row) {
        for (int32_t col = bound; col < image.cols() - bound; ++col) {
            if (this->mask()(row, col)) {
                const float response = ComputeResponseOfPixel(image, row, col);
                if (response > options().kMinValidResponse) {
                    this->candidates().insert(std::make_pair(response, Pixel(col, row)));
                }
            }
        }
    }
    return true;
}

}  // namespace feature_detector
