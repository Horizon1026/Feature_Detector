#include "feature_point_detector.h"
#include "slam_operations.h"

namespace FEATURE_DETECTOR {

bool FeaturePointDetector::DetectGoodFeatures(const Image &image,
                                         const uint32_t needed_feature_num,
                                         std::vector<Vec2> &features) {
    // Check input image.
    if (image.data() == nullptr) {
        return false;
    }

    // Detect all features to be candidates.
    candidates_.clear();
    if (SelectCandidates(image) == false) {
        return false;
    }

    // If there are already some detected features, do not detect new features besiding them.
    if (features.empty()) {
        mask_.setConstant(image.rows(), image.cols(), 1);
    } else {
        UpdateMaskByFeatures(image, features);
    }

    // Select good features by score from candidates.
    RETURN_FALSE_IF_FALSE(SelectGoodFeatures(image, needed_feature_num, features));
    return true;
}


void FeaturePointDetector::SparsifyFeatures(const std::vector<Vec2> &features,
                                       const int32_t image_rows,
                                       const int32_t image_cols,
                                       const uint8_t status_need_filter,
                                       const uint8_t status_after_filter,
                                       std::vector<uint8_t> &status) {
    if (features.size() != status.size()) {
        status.resize(features.size(), 1);
    }

    // Grid filter to make points sparsely.
    const float grid_row_step = image_rows / (options_.kGridFilterRowDivideNumber - 1);
    const float grid_col_step = image_cols / (options_.kGridFilterColDivideNumber - 1);
    mask_.setConstant(options_.kGridFilterRowDivideNumber,
                      options_.kGridFilterColDivideNumber,
                      1);
    for (uint32_t i = 0; i < features.size(); ++i) {
        const int32_t row = static_cast<int32_t>(features[i].y() / grid_row_step);
        const int32_t col = static_cast<int32_t>(features[i].x() / grid_col_step);
        if (mask_(row, col) && status[i] == status_need_filter) {
            mask_(row, col) = 0;
        } else if (!mask_(row, col) && status[i] == status_need_filter) {
            status[i] = status_after_filter;
        }
    }
}

bool FeaturePointDetector::SelectCandidates(const Image &image) {
    switch (options_.kMethod) {
        case HARRIS: {
            if (harris_.ComputeGradient(image) == false) {
                return false;
            }

            const int32_t bound = harris_.options().kHalfPatchSize;
            for (int32_t row = bound; row < image.rows() - bound; ++row) {
                for (int32_t col = bound; col < image.cols() - bound; ++col) {
                    const float response = harris_.ComputeResponse(image, row, col);
                    if (response > options_.kMinValidResponse) {
                        candidates_.insert(std::make_pair(response, Eigen::Matrix<int32_t, 2, 1>(col, row)));
                    }
                }
            }
            return true;
        }

        case SHI_TOMAS: {
            if (shi_tomas_.ComputeGradient(image) == false) {
                return false;
            }

            const int32_t bound = shi_tomas_.options().kHalfPatchSize;
            for (int32_t row = bound; row < image.rows() - bound; ++row) {
                for (int32_t col = bound; col < image.cols() - bound; ++col) {
                    const float response = shi_tomas_.ComputeResponse(image, row, col);
                    if (response > options_.kMinValidResponse) {
                        candidates_.insert(std::make_pair(response, Eigen::Matrix<int32_t, 2, 1>(col, row)));
                    }
                }
            }
            return true;
        }

        case FAST: {
            const int32_t bound = fast_.options().kHalfPatchSize;
            float offset = 1e-5f;
            for (int32_t row = bound; row < image.rows() - bound; ++row) {
                for (int32_t col = bound; col < image.cols() - bound; ++col) {
                    const float response = fast_.ComputeResponse(image, row, col) + offset;
                    if (response > options_.kMinValidResponse) {
                        candidates_.insert(std::make_pair(response, Eigen::Matrix<int32_t, 2, 1>(col, row)));
                    }

                    offset += 1e-5;
                }
            }
            return true;;
        }

        default:
            break;
    }

    return true;
}

bool FeaturePointDetector::SelectGoodFeatures(const Image &image,
                                         const uint32_t needed_feature_num,
                                         std::vector<Vec2> &features) {
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

void FeaturePointDetector::DrawRectangleInMask(const int32_t row,
                                          const int32_t col) {
    for (int32_t drow = - options_.kMinFeatureDistance; drow <= options_.kMinFeatureDistance; ++drow) {
        for (int32_t dcol = - options_.kMinFeatureDistance; dcol <= options_.kMinFeatureDistance; ++dcol) {
            const int32_t sub_row = drow + row;
            const int32_t sub_col = dcol + col;
            if (sub_row < 0 || sub_col < 0 || sub_row > mask_.rows() - 1 || sub_col > mask_.cols() - 1) {
                continue;
            }

            mask_(sub_row, sub_col) = 0;
        }
    }
}

void FeaturePointDetector::UpdateMaskByFeatures(const Image &image,
                                           const std::vector<Vec2> &features) {
    mask_.setConstant(image.rows(), image.cols(), 1);

    for (const auto &feature : features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col);
    }
}

}