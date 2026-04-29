#ifndef _FEATURE_POINT_HARRIS_DETECTOR_H_
#define _FEATURE_POINT_HARRIS_DETECTOR_H_

#include "feature_point_detector.h"

namespace feature_detector {

/* Class FeaturePointHarrisDetector Declaration. */
class FeaturePointHarrisDetector: public FeaturePointDetector {

public:
    struct SubOptions {
        float kAlpha = 0.04f;
        int32_t kHalfPatchSize = 1;
    };

public:
    FeaturePointHarrisDetector() = default;
    virtual ~FeaturePointHarrisDetector() = default;

    virtual std::string DetectorTypeName() const override { return "Harris"; }

private:
    virtual bool ComputeCandidates(const GrayImage &image) override;
    void ComputeHorizontalGradientSums(const GrayImage &image);
    void ComputeResponseMap();
    void PerformNMSAndExtractCandidates();

private:
    SubOptions sub_options_;
    MatImgF Ix_;
    MatImgF Iy_;
    MatImgF Ixx_;
    MatImgF Iyy_;
    MatImgF Ixy_;
    MatImgF Sxx_;
    MatImgF Syy_;
    MatImgF Sxy_;
    MatImgF tmp_;
    MatImgF responses_;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_POINT_HARRIS_DETECTOR_H_
