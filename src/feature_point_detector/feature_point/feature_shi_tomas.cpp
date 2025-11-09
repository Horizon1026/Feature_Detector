#include "feature_shi_tomas.h"

namespace feature_detector {

bool ShiTomasFeature::ComputeGradient(const GrayImage &image) {
    Ix_.setZero(image.rows(), image.cols());
    Iy_.setZero(image.rows(), image.cols());

    for (int32_t row = options().kHalfPatchSize; row < image.rows() - options().kHalfPatchSize; ++row) {
        for (int32_t col = options().kHalfPatchSize; col < image.cols() - options().kHalfPatchSize; ++col) {
            Ix_(row, col) = image.GetPixelValueNoCheck<float>(row, col + 1) - image.GetPixelValueNoCheck<float>(row, col - 1);
            Iy_(row, col) = image.GetPixelValueNoCheck<float>(row + 1, col) - image.GetPixelValueNoCheck<float>(row - 1, col);
        }
    }

    return true;
}

float ShiTomasFeature::ComputeResponse(const GrayImage &image, const int32_t row, const int32_t col) {
    Mat2 M = Mat2::Zero();
    int32_t cnt = 0;
    for (int32_t drow = -options().kHalfPatchSize; drow <= options().kHalfPatchSize; ++drow) {
        for (int32_t dcol = -options().kHalfPatchSize; dcol <= options().kHalfPatchSize; ++dcol) {
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

bool ShiTomasFeature::SelectAllCandidates(const GrayImage &image, const MatInt &mask, std::map<float, Pixel> &candidates) {
    if (ComputeGradient(image) == false) {
        return false;
    }

    const int32_t bound = options().kHalfPatchSize;
    for (int32_t row = bound; row < image.rows() - bound; ++row) {
        for (int32_t col = bound; col < image.cols() - bound; ++col) {
            if (mask(row, col)) {
                const float response = ComputeResponse(image, row, col);
                if (response > options().kMinValidResponse) {
                    candidates.insert(std::make_pair(response, Pixel(col, row)));
                }
            }
        }
    }
    return true;
}

}  // namespace feature_detector