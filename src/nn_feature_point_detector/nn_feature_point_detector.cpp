#include "nn_feature_point_detector.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"

namespace FEATURE_DETECTOR {

bool NNFeaturePointDetector::ReloadModel(const std::string &model_path) {
    try {
        nn_model_ = torch::jit::load(model_path);
        nn_model_.eval();
    } catch (const c10::Error &e) {
        ReportError("[NN Feature Detector] Failed to reload model from: " << model_path);
        return false;
    }

    return true;
}

NNFeaturePointDetector::NNFeaturePointDetector(const std::string &model_path) {
    ReloadModel(model_path);
}

void NNFeaturePointDetector::DrawRectangleInMask(const int32_t row, const int32_t col, const int32_t radius) {
    const int32_t row_start = std::max(0, row - radius);
    const int32_t row_end = std::min(static_cast<int32_t>(mask_.rows() - 1), row + radius);
    const int32_t col_start = std::max(0, col - radius);
    const int32_t col_end = std::min(static_cast<int32_t>(mask_.cols() - 1), col + radius);

    for (int32_t r = row_start; r <= row_end; ++r) {
        for (int32_t c = col_start; c <= col_end; ++c) {
            mask_(r, c) = 0;
        }
    }
}

void NNFeaturePointDetector::UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features) {
    for (const auto &feature : features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col, options_.kMinFeatureDistance);
    }
}

bool NNFeaturePointDetector::SelectKeypointCandidatesFromHeatMap() {
    RETURN_FALSE_IF(keypoints_heat_map_.size() == 0);
    RETURN_FALSE_IF(mask_.rows() != keypoints_heat_map_.rows() || mask_.cols() != keypoints_heat_map_.cols());
    candidates_.clear();

    for (int32_t row = 0; row < keypoints_heat_map_.rows(); ++row) {
        for (int32_t col = 0; col < keypoints_heat_map_.cols(); ++col) {
            CONTINUE_IF(!mask_(row, col));
            const float response = keypoints_heat_map_(row, col);
            if (response > options_.kMinResponse) {
                candidates_.emplace(response, Pixel(col, row));
            }
        }
    }

    return true;
}

bool NNFeaturePointDetector::SelectGoodFeaturesFromCandidates(const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    for (auto it = candidates_.crbegin(); it != candidates_.crend(); ++it) {
        const Pixel pixel = it->second;
        const int32_t row = pixel.y();
        const int32_t col = pixel.x();
        if (mask_(row, col)) {
            features.emplace_back(Vec2(pixel.x(), pixel.y()));
            RETURN_TRUE_IF(features.size() >= needed_feature_num);
            DrawRectangleInMask(row, col, options_.kMinFeatureDistance);
        }
    }

    return true;
}


} // End of namespace FEATURE_DETECTOR.
