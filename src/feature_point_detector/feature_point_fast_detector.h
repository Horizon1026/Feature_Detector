#ifndef _FEATURE_POINT_FAST_DETECTOR_H_
#define _FEATURE_POINT_FAST_DETECTOR_H_

#include "feature_point_detector.h"

namespace feature_detector {

class FeaturePointFastDetector : public FeaturePointDetector {

public:
    struct FeatureOptions {
        int32_t kN = 12;
        uint8_t kMinPixelDiffValue = 15;
    };

public:
    FeaturePointFastDetector() = default;
    virtual ~FeaturePointFastDetector() = default;

private:
    virtual bool ComputeCandidates(const GrayImage &image) override;
    float ComputeResponseOfPixel(const GrayImage &image, const int32_t row, const int32_t col);

private:
    FeatureOptions feature_options_;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_POINT_FAST_DETECTOR_H_
