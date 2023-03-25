#include "feature_detector.h"

#include "slam_operations.h"

namespace FEATURE_DETECTOR {

bool FeatureDetector::DetectGoodFeatures(const Image *image,
                                         const uint32_t needed_feature_num,
                                         std::vector<Vec2> &features) {
    if (image == nullptr) {
        return false;
    } else if (image->image_data() == nullptr) {
        return false;
    }

    candidates_.clear();
    if (SelectCandidates(image) == false) {
        return false;
    }

    features.clear();
    features.reserve(needed_feature_num);
    mask_.setConstant(image->rows(), image->cols(), 1);
    if (SelectGoodFeatures(image, needed_feature_num, features) == false) {
        return false;
    }

    return true;
}

bool FeatureDetector::SelectCandidates(const Image *image) {
    switch (options_.kMethod) {
        case HARRIS: {
            if (harris_.ComputeGradient(image) == false) {
                return false;
            }

            const int32_t bound = harris_.options().kHalfPatchSize;
            for (int32_t row = bound; row < image->rows() - bound; ++row) {
                for (int32_t col = bound; col < image->cols() - bound; ++col) {
                    const float response = harris_.ComputeResponse(image, row, col);
                    if (response > options_.kMinValidResponse) {
                        candidates_.insert(std::make_pair(response, Eigen::Matrix<int32_t, 2, 1>(col, row)));
                    }
                }
            }
            return true;;
        }

        case SHI_TOMAS: {
            if (shi_tomas_.ComputeGradient(image) == false) {
                return false;
            }

            const int32_t bound = shi_tomas_.options().kHalfPatchSize;
            for (int32_t row = bound; row < image->rows() - bound; ++row) {
                for (int32_t col = bound; col < image->cols() - bound; ++col) {
                    const float response = shi_tomas_.ComputeResponse(image, row, col);
                    if (response > options_.kMinValidResponse) {
                        candidates_.insert(std::make_pair(response, Eigen::Matrix<int32_t, 2, 1>(col, row)));
                    }
                }
            }
            return true;;
        }

        case FAST: {
            const int32_t bound = fast_.options().kHalfPatchSize;
            float offset = 1e-5f;
            for (int32_t row = bound; row < image->rows() - bound; ++row) {
                for (int32_t col = bound; col < image->cols() - bound; ++col) {
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

bool FeatureDetector::SelectGoodFeatures(const Image *image,
                                         const uint32_t needed_feature_num,
                                         std::vector<Vec2> &features) {
    for (auto it = candidates_.rbegin(); it != candidates_.rend(); ++it) {
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

void FeatureDetector::DrawRectangleInMask(const int32_t row,
                                          const int32_t col) {
    for (int32_t drow = - options_.kMinFeatureDistance; drow <= options_.kMinFeatureDistance; ++drow) {
        for (int32_t dcol = - options_.kMinFeatureDistance; dcol <= options_.kMinFeatureDistance; ++dcol) {
            const int32_t sub_row = drow + row;
            const int32_t sub_col = dcol + col;
            if (sub_row < 0 || sub_col < 0 || sub_row > mask_.rows() - 2 || sub_col > mask_.cols() - 2) {
                continue;
            }

            mask_(sub_row, sub_col) = 0;
        }
    }
}

}
