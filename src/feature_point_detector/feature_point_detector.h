#ifndef _FEATURE_POINT_DETECTOR_H_
#define _FEATURE_POINT_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace feature_detector {

class FeaturePointDetector {

public:
    struct Options {
        int32_t kMinFeatureDistance = 15;
        int32_t kGridFilterRowDivideNumber = 12;
        int32_t kGridFilterColDivideNumber = 12;
        float kMinValidResponse = 0.1f;
    };

public:
    FeaturePointDetector() = default;
    virtual ~FeaturePointDetector() = default;
    FeaturePointDetector(const FeaturePointDetector &detecor) = delete;

    bool DetectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features);

    void SparsifyFeatures(const std::vector<Vec2> &features, const int32_t image_rows, const int32_t image_cols, const uint8_t status_need_filter,
                          const uint8_t status_after_filter, std::vector<uint8_t> &status);

    // Reference for member variables.
    Options &options() { return options_; }
    std::map<float, Pixel> &candidates() { return candidates_; }
    MatInt &mask() { return mask_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }
    const std::map<float, Pixel> &candidates() const { return candidates_; }
    const MatInt &mask() const { return mask_; }

private:
    virtual bool ComputeCandidates(const GrayImage &image) = 0;
    bool SelectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features);
    void DrawRectangleInMask(const int32_t row, const int32_t col);
    void UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features);

private:
    Options options_;
    std::map<float, Pixel> candidates_;
    MatInt mask_;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_POINT_DETECTOR_H_
