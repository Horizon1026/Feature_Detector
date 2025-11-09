#ifndef _FEATURE_POINT_HARRIS_DETECTOR_H_
#define _FEATURE_POINT_HARRIS_DETECTOR_H_

#include "feature_point_detector.h"

namespace feature_detector {

class FeaturePointHarrisDetector: public FeaturePointDetector {

public:
    struct FeatureOptions {
        float kAlpha = 0.04f;
        int32_t kHalfPatchSize = 1;
    };

public:
    FeaturePointHarrisDetector() = default;
    virtual ~FeaturePointHarrisDetector() = default;

private:
    virtual bool ComputeCandidates(const GrayImage &image) override;
    bool ComputeGradient(const GrayImage &image);
    float ComputeResponseOfPixel(const GrayImage &image, const int32_t row, const int32_t col);

private:
    FeatureOptions feature_options_;
    MatImgF Ix_;
    MatImgF Iy_;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_POINT_HARRIS_DETECTOR_H_
