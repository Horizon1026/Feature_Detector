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
    RETURN_TRUE_IF(needed_feature_num == 0);

    // Compute minimal number of pixels in a region, which can give a meaningful event.
    const float p = options_.kMinToleranceAngleResidualInRad / kPai;
    const float log_NT = 5.0f * (std::log10(double(image.cols())) + std::log10(double(image.rows()))) / 2.0f + std::log10(11.0f);
    const uint32_t min_region_size = static_cast<uint32_t>(- log_NT / std::log10(p));

    RETURN_FALSE_IF_FALSE(ComputeLineLevelAngleMap(image));

    // Search for line segments.
    RegionParam region;
    rectangles_.clear();
    for (const auto &sorted_pixel : sorted_pixels_) {
        // Try to grow new region from seed pixel. Clear pixels consist of it if this region is invalid.
        CONTINUE_IF(!sorted_pixel->is_valid || sorted_pixel->is_used);
        GrowRegion(*sorted_pixel, region);
        if (region.pixels.size() < min_region_size) {
            for (auto &pixel : region.pixels) {
                pixel->is_used = false;
            }
            continue;
        }

        // Convert region to rectangle.
        RectangleParam rectangle = ConvertRegionToRectangle(region);
        CONTINUE_IF(rectangle.length < options_.kMinValidLineLengthInPixel ||
            rectangle.inlier_ratio < options_.kMaxToleranceInlierRation);

        // Compensate the offset.
        rectangle.start_point += Vec2::Constant(0.5f);
        rectangle.end_point += Vec2::Constant(0.5f);
        rectangles_.emplace_back(rectangle);
    }

    // Output.
    features.clear();
    for (const auto &rect : rectangles_) {
        features.emplace_back(Vec4(rect.start_point.x(), rect.start_point.y(), rect.end_point.x(), rect.end_point.y()));
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

void FeatureLineDetector::GrowRegion(PixelParam &seed_pixel, RegionParam &region) {
    candidates_.Clear();
    visited_pixels_.Clear();
    visited_pixels_.PushBack(&seed_pixel);
    seed_pixel.is_occupied = true;

    // Initialize region with seed pixel.
    region.pixels.clear();
    region.angle = seed_pixel.line_level_angle;
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
        const float angle_residual = Utility::AngleDiffInRad(region.angle, pixel_ptr->line_level_angle);
        if (std::fabs(angle_residual) > options_.kMinToleranceAngleResidualInRad) {
            continue;
        }
        // Add it into region if aligned. Recompute angle of this region.
        sum_dx += std::cos(pixel_ptr->line_level_angle);
        sum_dy += std::sin(pixel_ptr->line_level_angle);
        region.angle = std::atan2(sum_dy, sum_dx);
        region.pixels.emplace_back(pixel_ptr);
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

FeatureLineDetector::RectangleParam FeatureLineDetector::ConvertRegionToRectangle(const RegionParam &region) {
    RectangleParam rect;
    // Compute centroid of rectangle.
    float sum_weight = 0.0f;
    for (const auto &pixel : region.pixels) {
        rect.center_point.x() += static_cast<float>(pixel->col) * pixel->gradient_norm;
        rect.center_point.y() += static_cast<float>(pixel->row) * pixel->gradient_norm;
        sum_weight += pixel->gradient_norm;
    }
    if (sum_weight == 0) {
        return rect;
    }
    rect.center_point /= sum_weight;

    // Compute angle of rectangle.
    float Ixx = 0.0f;
    float Iyy = 0.0f;
    float Ixy = 0.0f;
    for (const auto &pixel : region.pixels) {
        const float dx = pixel->col - rect.center_point.x();
        const float dy = pixel->row - rect.center_point.y();
        Ixx += dy * dy * pixel->gradient_norm;
        Iyy += dx * dx * pixel->gradient_norm;
        Ixy -= dx * dy * pixel->gradient_norm;
    }
    if (Ixx == 0 || Iyy == 0 || Ixy == 0) {
        return rect;
    }
    const float smallest_eiven_value = 0.5f * (Ixx + Iyy - std::sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0f * Ixy * Ixy));
    rect.angle = std::fabs(Ixx) > std::fabs(Iyy) ? std::atan2(smallest_eiven_value - Ixx, Ixy) : std::atan2(Ixy, smallest_eiven_value - Iyy);
    if (std::fabs(Utility::AngleDiffInRad(rect.angle, region.angle)) > options_.kMinToleranceAngleResidualInRad) {
        rect.angle += kPai;
        if (rect.angle >= kPai) {
            rect.angle -= k2Pai;
        }
    }
    rect.dir_vector = Vec2(std::cos(rect.angle), std::sin(rect.angle));
    rect.dir_vector.x() = std::cos(rect.angle);
    rect.dir_vector.y() = std::sin(rect.angle);

    // Compute length and width of rectangle.
    Vec2 length_range = Vec2::Zero();
    Vec2 width_range = Vec2::Zero();
    for (const auto &pixel : region.pixels) {
        const float region_dx = pixel->col - rect.center_point.x();
        const float region_dy = pixel->row - rect.center_point.y();
        const float length = region_dx * rect.dir_vector.x() + region_dy * rect.dir_vector.y();
        const float width = - region_dx * rect.dir_vector.y() + region_dy * rect.dir_vector.x();
        length_range(0) = std::min(length_range(0), length);
        length_range(1) = std::max(length_range(1), length);
        width_range(0) = std::min(width_range(0), width);
        width_range(1) = std::max(width_range(1), width);
    }

    // Update paremeters of rectangle. And compute inlier ratio.
    rect.start_point = rect.center_point + length_range(0) * rect.dir_vector;
    rect.end_point = rect.center_point + length_range(1) * rect.dir_vector;
    rect.length = length_range(1) - length_range(0);
    rect.width = width_range(1) - width_range(0);
    // Lenght and width should be at least 1 pixel.
    rect.length = std::max(rect.length, 1.0f);
    rect.width = std::max(rect.width, 1.0f);
    const float area_size = (length_range(1) - length_range(0)) * rect.width;
    rect.inlier_ratio = static_cast<float>(region.pixels.size()) / area_size;
    return rect;
}

}
