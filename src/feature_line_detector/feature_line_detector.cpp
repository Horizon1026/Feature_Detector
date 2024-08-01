#include "feature_line_detector.h"
#include "math_kinematics.h"
#include "slam_operations.h"
#include "log_report.h"

namespace FEATURE_DETECTOR {

FeatureLineDetector::FeatureLineDetector() {
    sorted_pixels_.clear();
    sorted_pixels_.reserve(76800);
}

bool FeatureLineDetector::DetectGoodFeatures(const GrayImage &image,
                                             const uint32_t needed_feature_num,
                                             std::vector<Vec4> &features) {
    RETURN_FALSE_IF(image.data() == nullptr || image.rows() < 2 || image.cols() < 2);
    features.clear();
    RETURN_TRUE_IF(needed_feature_num == 0);

    RETURN_FALSE_IF_FALSE(ComputeLineLevelAngleMap(image));

    return true;
}

bool FeatureLineDetector::ComputeLineLevelAngleMap(const GrayImage &image)
{
    pixels_.resize(image.rows(), image.cols());

    for (int32_t col = 0; col < image.cols() - 1; ++col) {
        for (int32_t row = 0; row < image.rows() - 1; ++row) {
            // Compute pixel gradient.
            const int32_t pixel_ad = static_cast<int32_t>(image.GetPixelValueNoCheck(row + 1, col + 1)) -
                static_cast<int32_t>(image.GetPixelValueNoCheck(row, col));
            const int32_t pixel_bc = static_cast<int32_t>(image.GetPixelValueNoCheck(row, col + 1)) -
                static_cast<int32_t>(image.GetPixelValueNoCheck(row + 1, col));
            const float gradient_x = static_cast<float>(pixel_ad + pixel_bc) / 2.0f;
            const float gradient_y = static_cast<float>(pixel_ad - pixel_bc) / 2.0f;
            pixels_(row, col).gradient_norm = std::sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
            pixels_(row, col).is_valid = pixels_(row, col).gradient_norm > options_.kMinValidGradientNorm;
            if (pixels_(row, col).is_valid) {
                pixels_(row, col).line_level_angle = std::atan2(gradient_x, - gradient_y) * kDegToRad;
                sorted_pixels_.emplace_back(&pixels_(row, col));
            }
        }
    }

    // Sort pixels by gradient norm.
    std::sort(sorted_pixels_.begin(), sorted_pixels_.end(), [&](PixelParam *pixel1, PixelParam * pixel2) {
        return pixel1->gradient_norm < pixel2->gradient_norm;
    });

    return true;
}

}
