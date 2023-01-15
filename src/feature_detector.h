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
        int32_t kMinFeatureDistance = 20;
        FeatureDetectMethod kMethod = HARRIS;
    };

public:
    explicit FeatureDetector() = default;
    virtual ~FeatureDetector() = default;

    FeatureDetectOptions &options() { return options_; }

    bool DetectGoodFeatures(const Image *image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec2> &features);

private:
    bool SelectCandidates(const Image *image);

    bool SelectGoodFeatures(const Image *image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec2> &features);

    void DrawRectangleInMask(const int32_t row,
                             const int32_t col);

private:
    std::map<float, Pixel> candidates_;
    FeatureDetectOptions options_;
    Mat mask_;

    HarrisFeature harris_;
    ShiTomasFeature shi_tomas_;
    FastFeature fast_;

};

}

#endif // end of _FEATURE_DETECTOR_H_
