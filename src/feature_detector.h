#ifndef _FEATURE_DETECTOR_H_
#define _FEATURE_DETECTOR_H_

#include "datatype_basic.h"
#include "datatype_image.h"

#include "feature_harris.h"
#include "feature_shi_tomas.h"
#include "feature_fast.h"

namespace FEATURE_DETECTOR {

class FeatureDetector {

public:
    enum FeatureDetectMethod: uint8_t {
        FAST = 0,
        HARRIS,
        SHI_TOMAS,
        ORB,
    };

    struct FeatureDetectOptions {
        float kMinValidResponse = 0.1f;
        int32_t kMinFeatureDistance = 15;
        FeatureDetectMethod kMethod = HARRIS;
        int32_t kGridFilterRowDivideNumber = 12;
        int32_t kGridFilterColDivideNumber = 12;
    };

public:
    explicit FeatureDetector() = default;
    virtual ~FeatureDetector() = default;

    FeatureDetectOptions &options() { return options_; }

    bool DetectGoodFeatures(const Image &image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec2> &features);

    void SparsifyFeatures(const std::vector<Vec2> &features,
                          const int32_t image_rows,
                          const int32_t image_cols,
                          const uint8_t status_need_filter,
                          const uint8_t status_after_filter,
                          std::vector<uint8_t> &status);

private:
    bool SelectCandidates(const Image &image);

    bool SelectGoodFeatures(const Image &image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec2> &features);

    void DrawRectangleInMask(const int32_t row,
                             const int32_t col);

    void UpdateMaskByFeatures(const Image &image,
                              const std::vector<Vec2> &features);

private:
    std::map<float, Pixel> candidates_;
    FeatureDetectOptions options_;
    MatInt mask_;

    HarrisFeature harris_;
    ShiTomasFeature shi_tomas_;
    FastFeature fast_;

};

}

#endif // end of _FEATURE_DETECTOR_H_
