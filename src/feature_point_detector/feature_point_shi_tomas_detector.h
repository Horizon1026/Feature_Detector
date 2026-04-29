#ifndef _FEATURE_POINT_SHI_TOMAS_DETECTOR_H_
#define _FEATURE_POINT_SHI_TOMAS_DETECTOR_H_

#include "feature_point_detector.h"

namespace feature_detector {

/* Class FeaturePointShiTomasDetector Declaration. */
class FeaturePointShiTomasDetector: public FeaturePointDetector {

public:
    struct SubOptions {
        int32_t kHalfPatchSize = 1;
    };

public:
    FeaturePointShiTomasDetector() = default;
    virtual ~FeaturePointShiTomasDetector() = default;

    virtual std::string DetectorTypeName() const override { return "Shi-Tomas"; }

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

#endif  // end of _FEATURE_POINT_SHI_TOMAS_DETECTOR_H_
