#include "feature_point_harris_detector.h"

namespace feature_detector {

bool FeaturePointHarrisDetector::ComputeCandidates(const GrayImage &image) {
    RETURN_FALSE_IF(image.data() == nullptr);
    // Horizontal pass: compute horizontal sums of gradients
    ComputeHorizontalGradientSums(image);
    // Vertical pass: cache-friendly sliding window and response calculation
    ComputeResponseMap();
    // Simple NMS and candidates extraction
    PerformNMSAndExtractCandidates();

    return true;
}

void FeaturePointHarrisDetector::ComputeHorizontalGradientSums(const GrayImage &image) {
    const int32_t rows = image.rows();
    const int32_t cols = image.cols();
    const int32_t half_size = sub_options_.kHalfPatchSize;
    const int32_t patch_size = 2 * half_size + 1;

    tmp_.resize(rows, 3 * cols);
    const uint8_t *data = image.data();

    std::vector<float> row_ixx(cols, 0.0f);
    std::vector<float> row_iyy(cols, 0.0f);
    std::vector<float> row_ixy(cols, 0.0f);

    for (int32_t r = 1; r < rows - 1; ++r) {
        const uint8_t *row_ptr = data + r * cols;
        const uint8_t *prev_row_ptr = row_ptr - cols;
        const uint8_t *next_row_ptr = row_ptr + cols;

        for (int32_t c = 1; c < cols - 1; ++c) {
            const float ix = static_cast<float>(row_ptr[c + 1]) - static_cast<float>(row_ptr[c - 1]);
            const float iy = static_cast<float>(next_row_ptr[c]) - static_cast<float>(prev_row_ptr[c]);
            row_ixx[c] = ix * ix;
            row_iyy[c] = iy * iy;
            row_ixy[c] = ix * iy;
        }

        float *tmp_row = tmp_.data() + r * 3 * cols;
        float sum_xx = 0;
        float sum_yy = 0;
        float sum_xy = 0;
        for (int32_t c = 0; c < patch_size; ++c) {
            sum_xx += row_ixx[c];
            sum_yy += row_iyy[c];
            sum_xy += row_ixy[c];
        }
        tmp_row[half_size * 3] = sum_xx;
        tmp_row[half_size * 3 + 1] = sum_yy;
        tmp_row[half_size * 3 + 2] = sum_xy;
        for (int32_t c = half_size + 1; c < cols - half_size; ++c) {
            sum_xx += row_ixx[c + half_size] - row_ixx[c - half_size - 1];
            sum_yy += row_iyy[c + half_size] - row_iyy[c - half_size - 1];
            sum_xy += row_ixy[c + half_size] - row_ixy[c - half_size - 1];
            tmp_row[c * 3] = sum_xx;
            tmp_row[c * 3 + 1] = sum_yy;
            tmp_row[c * 3 + 2] = sum_xy;
        }
    }
}

void FeaturePointHarrisDetector::ComputeResponseMap() {
    const int32_t rows = tmp_.rows();
    const int32_t cols = tmp_.cols() / 3;
    const int32_t half_size = sub_options_.kHalfPatchSize;
    const int32_t patch_size = 2 * half_size + 1;
    const float inv_cnt = 1.0f / static_cast<float>(patch_size * patch_size);
    const float inv_cnt2 = inv_cnt * inv_cnt;

    responses_.resize(rows, cols);
    responses_.setZero();

    std::vector<float> sum_xx_vec(cols, 0.0f);
    std::vector<float> sum_yy_vec(cols, 0.0f);
    std::vector<float> sum_xy_vec(cols, 0.0f);

    for (int32_t r = 1; r < 1 + patch_size && r < rows; ++r) {
        float *tmp_row = tmp_.data() + r * 3 * cols;
        for (int32_t c = half_size; c < cols - half_size; ++c) {
            sum_xx_vec[c] += tmp_row[c * 3];
            sum_yy_vec[c] += tmp_row[c * 3 + 1];
            sum_xy_vec[c] += tmp_row[c * 3 + 2];
        }
    }

    const int32_t bound = half_size + 1;
    for (int32_t r = bound; r < rows - bound; ++r) {
        float *res_row = responses_.data() + r * cols;
        for (int32_t c = bound; c < cols - bound; ++c) {
            if (this->mask()(r, c)) {
                const float sxx = sum_xx_vec[c];
                const float syy = sum_yy_vec[c];
                const float trace = sxx + syy;
                if (trace * trace * 0.21f * inv_cnt2 > options().kMinValidResponse) {
                    const float sxy = sum_xy_vec[c];
                    const float res = (sxx * syy - sxy * sxy - sub_options_.kAlpha * trace * trace) * inv_cnt2;
                    if (res > options().kMinValidResponse) {
                        res_row[c] = res;
                    }
                }
            }
        }

        if (r + half_size + 1 < rows - 1) {
            float *tmp_next = tmp_.data() + (r + half_size + 1) * 3 * cols;
            float *tmp_prev = tmp_.data() + (r - half_size) * 3 * cols;
            for (int32_t c = half_size; c < cols - half_size; ++c) {
                sum_xx_vec[c] += tmp_next[c * 3] - tmp_prev[c * 3];
                sum_yy_vec[c] += tmp_next[c * 3 + 1] - tmp_prev[c * 3 + 1];
                sum_xy_vec[c] += tmp_next[c * 3 + 2] - tmp_prev[c * 3 + 2];
            }
        }
    }
}

void FeaturePointHarrisDetector::PerformNMSAndExtractCandidates() {
    const int32_t rows = responses_.rows();
    const int32_t cols = responses_.cols();
    const int32_t bound = sub_options_.kHalfPatchSize + 1;
    for (int32_t r = bound; r < rows - bound; ++r) {
        float *res_row = responses_.data() + r * cols;
        float *res_prev = res_row - cols;
        float *res_next = res_row + cols;
        for (int32_t c = bound; c < cols - bound; ++c) {
            const float &res = res_row[c];
            CONTINUE_IF(res <= options().kMinValidResponse);
            if (res > res_row[c - 1] && res > res_row[c + 1] &&
                res > res_prev[c] && res > res_next[c]) {
                this->candidates().emplace_back(res, Pixel(c, r));
            }
        }
    }
}

}  // namespace feature_detector
