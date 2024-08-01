#ifndef _FEATURE_LINE_DETECTOR_H_
#define _FEATURE_LINE_DETECTOR_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class FeatureLineDetector {

public:
    struct PixelParam {
        float line_level_angle = 0.0f;
        float gradient_norm = 0.0f;
        bool is_used = false;
    };

public:
    FeatureLineDetector() = default;
    virtual ~FeatureLineDetector() = default;

public:
    bool DetectGoodFeatures(const GrayImage &image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec4> &features);

    // Reference for member variables.
    // Const reference for member variables.

private:
    bool ComputeLineLevelAngleMap(const GrayImage &image);

private:
    Eigen::Matrix<PixelParam, Eigen::Dynamic, Eigen::Dynamic> pixels_;
    std::array<PixelParam *, 1024> sorted_pixels_;

};

}

#endif // end of _FEATURE_LINE_DETECTOR_H_
