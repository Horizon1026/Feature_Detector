#include "feature_fast.h"

namespace FEATURE_DETECTOR {

constexpr int32_t fast_indice[][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3} };

float FastFeature::ComputeResponse(const Image &image,
                                   const int32_t row,
                                   const int32_t col) {

    int32_t pixel_value = image.GetPixelValueNoCheck<int32_t>(row, col);
    int32_t max_pixel_value = pixel_value + options_.kMinPixelDiffValue;
    int32_t min_pixel_value = pixel_value - options_.kMinPixelDiffValue;

    int32_t larger_cnt = 0;
    int32_t smaller_cnt = 0;

    // If Fast-12 or more, it can be precheck if it can be FAST corner.
    if (options_.kN >= 12) {
        int32_t idx[4] = {0, 4, 8, 12};

        for (uint32_t i = 0; i < 4; ++i) {
            int32_t pixel_arounded_value = image.GetPixelValueNoCheck<int32_t>(row + fast_indice[idx[i]][1], col + fast_indice[idx[i]][0]);

            if (pixel_arounded_value > max_pixel_value) {
                ++larger_cnt;
                smaller_cnt = 0;
            } else if (pixel_arounded_value < min_pixel_value) {
                ++smaller_cnt;
                larger_cnt = 0;
            } else {
                smaller_cnt = 0;
                larger_cnt = 0;
            }
        }

        // It cannot be a FAST corner.
        if (smaller_cnt < 3 && larger_cnt < 3) {
            return false;
        }
    }

    std::vector<int32_t> compare_results(16, 0);
    for (uint32_t i = 0; i < 16; ++i) {
        int32_t pixel_arounded_value = image.GetPixelValueNoCheck<int32_t>(row + fast_indice[i][1], col + fast_indice[i][0]);
        if (pixel_arounded_value > max_pixel_value) {
            compare_results[i] = 1;
        } else if (pixel_arounded_value < min_pixel_value) {
            compare_results[i] = -1;
        }
    }

    // Check whether it is FAST corner.
    smaller_cnt = 0;
    larger_cnt = 0;
    int32_t best_cnt = 0;
    for (int32_t k = 0; k < 2 && best_cnt < 16; ++k) {
        for (uint32_t i = 0; i < compare_results.size(); ++i) {
            if (compare_results[i] == 1) {
                larger_cnt++;
                smaller_cnt = 0;
            } else if (compare_results[i] == -1) {
                smaller_cnt++;
                larger_cnt = 0;
            } else {
                smaller_cnt = 0;
                larger_cnt = 0;
            }

            if (larger_cnt > best_cnt) {
                best_cnt = larger_cnt;
            }
            if (smaller_cnt > best_cnt) {
                best_cnt = smaller_cnt;
            }
        }
    }

    response_ = static_cast<float>(best_cnt);
    return response_;
}

}
