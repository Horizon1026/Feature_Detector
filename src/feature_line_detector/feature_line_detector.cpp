#include "feature_line_detector.h"
#include "slam_operations.h"
#include "log_report.h"

namespace FEATURE_DETECTOR {

FeatureLineDetector::FeatureLineDetector() {
    sorted_pixels_.clear();
    sorted_pixels_.reserve(10000);
}

bool FeatureLineDetector::DetectGoodFeatures(const GrayImage &image,
                                             const uint32_t needed_feature_num,
                                             std::vector<Vec4> &features) {
    // Validate parameters.
    RETURN_FALSE_IF(image.data() == nullptr || image.rows() < 2 || image.cols() < 2);
    features.clear();
    RETURN_TRUE_IF(needed_feature_num == 0);

    // Compute minimal number of pixels in a region, which can give a meaningful event.
    const float p = options_.kMinToleranceAngleResidualInRad / kPai;
    const float log_NT = 5.0f * (std::log10(double(image.cols())) + std::log10(double(image.rows()))) / 2.0f + std::log10(11.0f);
    const uint32_t min_region_size = static_cast<uint32_t>(- log_NT / std::log10(p));

    RETURN_FALSE_IF_FALSE(ComputeLineLevelAngleMap(image));

    // Search for line segments.
    for (const auto &sorted_pixel : sorted_pixels_) {
        CONTINUE_IF(sorted_pixel->is_used || !sorted_pixel->is_valid);
        GrowRegion(*sorted_pixel);
        if (region_.pixels.size() < min_region_size) {
            for (auto &pixel : region_.pixels) {
                pixel->is_used = false;
            }
            continue;
        }
    }

    return true;
}

bool FeatureLineDetector::ComputeLineLevelAngleMap(const GrayImage &image)
{
    // The bottom-right boundary of image will be invalid, because gradient is invalid.
    pixels_.resize(image.rows() - 1, image.cols() - 1);
    for (int32_t i = 0; i < pixels_.rows(); ++i) {
        pixels_(i, 0).row = i;
        pixels_(i, pixels_.cols() - 1).row = i;
        pixels_(i, pixels_.cols() - 1).col = pixels_.cols() - 1;
    }
    for (int32_t i = 0; i < pixels_.cols(); ++i) {
        pixels_(0, 1).col = i;
        pixels_(pixels_.rows() - 1, i).col = i;
        pixels_(pixels_.rows() - 1, i).row = pixels_.rows() - 1;
    }

    // The boundary of 'pixels_' should be set to be invalid. Then there is no need to check pixel location overflow.
    for (int32_t col = 1; col < image.cols() - 2; ++col) {
        for (int32_t row = 1; row < image.rows() - 2; ++row) {
            pixels_(row, col).row = row;
            pixels_(row, col).col = col;
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
                pixels_(row, col).line_level_angle = std::atan2(gradient_x, - gradient_y);
                sorted_pixels_.emplace_back(&pixels_(row, col));
            }
        }
    }

    // Sort pixels by gradient norm from large to small.
    std::sort(sorted_pixels_.begin(), sorted_pixels_.end(), [&](PixelParam *pixel1, PixelParam * pixel2) {
        return pixel1->gradient_norm > pixel2->gradient_norm;
    });

    return true;
}

void FeatureLineDetector::GrowRegion(PixelParam &seed_pixel) {
    candidates_.Clear();
    visited_pixels_.Clear();
    visited_pixels_.PushBack(&seed_pixel);
    seed_pixel.is_occupied = true;

    // Initialize region with seed pixel.
    region_.pixels.clear();
    region_.angle = seed_pixel.line_level_angle;
    float sum_dx = std::cos(seed_pixel.line_level_angle);
    float sum_dy = std::sin(seed_pixel.line_level_angle);

    // Add its neighbour.
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row - 1, seed_pixel.col - 1));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row - 1, seed_pixel.col));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row - 1, seed_pixel.col + 1));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row, seed_pixel.col - 1));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row, seed_pixel.col + 1));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row + 1, seed_pixel.col - 1));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row + 1, seed_pixel.col));
    TryToAddPixelIntoCandidates(pixels_(seed_pixel.row + 1, seed_pixel.col + 1));

    // Grow region.
    while (!candidates_.Empty()) {
        // Got the next pixel.
        const auto &pixel_ptr = candidates_.Front();
        candidates_.PopFront();
        visited_pixels_.PushBack(pixel_ptr);
        // Check if it is aligned with this region.
        const float angle_residual = Utility::AngleDiffInRad(region_.angle, pixel_ptr->line_level_angle);
        if (std::fabs(angle_residual) > options_.kMinToleranceAngleResidualInRad) {
            continue;
        }
        // Add it into region if aligned. Recompute angle of this region.
        sum_dx += std::cos(pixel_ptr->line_level_angle);
        sum_dy += std::sin(pixel_ptr->line_level_angle);
        region_.angle = std::atan2(sum_dy, sum_dx);
        region_.pixels.emplace_back(pixel_ptr);
        pixel_ptr->is_used = true;
        // Add its neighbour.
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row - 1, pixel_ptr->col - 1));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row - 1, pixel_ptr->col));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row - 1, pixel_ptr->col + 1));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row, pixel_ptr->col - 1));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row, pixel_ptr->col + 1));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row + 1, pixel_ptr->col - 1));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row + 1, pixel_ptr->col));
        TryToAddPixelIntoCandidates(pixels_(pixel_ptr->row + 1, pixel_ptr->col + 1));
    }

    // Clear flag of occupied.
    while (!visited_pixels_.Empty()) {
        visited_pixels_.Front()->is_occupied = false;
        visited_pixels_.PopFront();
    }
}

void FeatureLineDetector::TryToAddPixelIntoCandidates(PixelParam &neighbour) {
    if (!neighbour.is_occupied && !neighbour.is_used && neighbour.is_valid) {
        neighbour.is_occupied = true;
        candidates_.PushBack(&neighbour);
    }
}

}
