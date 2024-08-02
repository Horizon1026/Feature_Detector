#ifndef _FEATURE_LINE_DETECTOR_H_
#define _FEATURE_LINE_DETECTOR_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "circular_buffer.h"
#include "math_kinematics.h"

namespace FEATURE_DETECTOR {

class FeatureLineDetector {

public:
    struct PixelParam {
        int32_t row = 0;
        int32_t col = 0;
        float line_level_angle = 0.0f;
        float gradient_norm = 0.0f;
        bool is_valid = false;
        bool is_used = false;
    };

    struct RegionParam {
        std::vector<PixelParam *> pixels;
        float angle = 0.0f;
    };

    struct Options {
        float kMinValidGradientNorm = 5.0f;
        float kMinToleranceAngleResidualInRad = 22.5f * kDegToRad;
    };

public:
    FeatureLineDetector();
    virtual ~FeatureLineDetector() = default;

    bool DetectGoodFeatures(const GrayImage &image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec4> &features);

    // Reference for member variables.
    Options &options() { return options_; }
    Eigen::Matrix<PixelParam, Eigen::Dynamic, Eigen::Dynamic> &pixels() { return pixels_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }
    const Eigen::Matrix<PixelParam, Eigen::Dynamic, Eigen::Dynamic> &pixels() const { return pixels_; }

private:
    bool ComputeLineLevelAngleMap(const GrayImage &image);
    int32_t GrowRegion(int32_t row, int32_t col);

private:
    Options options_;

    Eigen::Matrix<PixelParam, Eigen::Dynamic, Eigen::Dynamic> pixels_;
    std::vector<PixelParam *> sorted_pixels_;
    RegionParam region_;
    CircularBuffer<Pixel, 10000> candidates_;

};

}

#endif // end of _FEATURE_LINE_DETECTOR_H_
