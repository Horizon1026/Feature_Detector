#include "feature_point_detector.h"

namespace feature_detector {

/* Class Feature Point Detector Definition. */
bool FeaturePointDetector::DetectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    // Check input image.
    RETURN_FALSE_IF(image.data() == nullptr);

    // If there are already some detected features, do not detect new features besiding them.
    if (features.empty()) {
        mask_.setConstant(image.rows(), image.cols(), 1);
    } else {
        UpdateMaskByFeatures(image, features);
    }

    // Detect all features to be candidates.
    candidates_.clear();
    RETURN_FALSE_IF_FALSE(ComputeCandidates(image));

    // Select good features by score from candidates.
    RETURN_FALSE_IF_FALSE(SelectGoodFeatures(image, needed_feature_num, features));
    return true;
}

void FeaturePointDetector::SparsifyFeatures(const std::vector<Vec2> &features, const int32_t image_rows, const int32_t image_cols,
                                                         const uint8_t status_need_filter, const uint8_t status_after_filter, std::vector<uint8_t> &status) {
    if (features.size() != status.size()) {
        status.resize(features.size(), 1);
    }

    // Grid filter to make points sparsely.
    const float grid_row_step = image_rows / (options_.kGridFilterRowDivideNumber - 1);
    const float grid_col_step = image_cols / (options_.kGridFilterColDivideNumber - 1);
    mask_.setConstant(options_.kGridFilterRowDivideNumber, options_.kGridFilterColDivideNumber, 1);
    for (uint32_t i = 0; i < features.size(); ++i) {
        const int32_t row = static_cast<int32_t>(features[i].y() / grid_row_step);
        const int32_t col = static_cast<int32_t>(features[i].x() / grid_col_step);

        if (row < 0 || row > mask_.rows() - 1 || col < 0 || col > mask_.cols() - 1) {
            status[i] = status_after_filter;
            continue;
        }

        if (mask_(row, col) && status[i] == status_need_filter) {
            mask_(row, col) = 0;
        } else if (!mask_(row, col) && status[i] == status_need_filter) {
            status[i] = status_after_filter;
        }
    }
}

bool FeaturePointDetector::SelectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    for (auto it = candidates_.crbegin(); it != candidates_.crend(); ++it) {
        const Pixel pixel = it->second;
        const int32_t row = pixel.y();
        const int32_t col = pixel.x();
        if (mask_(row, col)) {
            features.emplace_back(Vec2(pixel.x(), pixel.y()));

            if (features.size() >= needed_feature_num) {
                return true;
            }

            DrawRectangleInMask(row, col);
        }
    }

    return true;
}

void FeaturePointDetector::DrawRectangleInMask(const int32_t row, const int32_t col) {
    for (int32_t drow = -options_.kMinFeatureDistance; drow <= options_.kMinFeatureDistance; ++drow) {
        for (int32_t dcol = -options_.kMinFeatureDistance; dcol <= options_.kMinFeatureDistance; ++dcol) {
            const int32_t sub_row = drow + row;
            const int32_t sub_col = dcol + col;
            if (sub_row < 0 || sub_col < 0 || sub_row > mask_.rows() - 1 || sub_col > mask_.cols() - 1) {
                continue;
            }

            mask_(sub_row, sub_col) = 0;
        }
    }
}

void FeaturePointDetector::UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features) {
    mask_.setConstant(image.rows(), image.cols(), 1);

    for (const auto &feature: features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col);
    }
}

}
