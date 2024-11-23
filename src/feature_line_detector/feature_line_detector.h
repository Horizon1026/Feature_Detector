#ifndef _FEATURE_LINE_DETECTOR_H_
#define _FEATURE_LINE_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "circular_buffer.h"
#include "slam_basic_math.h"

namespace FEATURE_DETECTOR {

class FeatureLineDetector {

public:
    struct PixelParam {
        int32_t row = 0;
        int32_t col = 0;
        float line_level_angle = 0.0f;
        float gradient_norm = 0.0f;
        bool is_valid = false;  // Valid when gradient norm is large enough.
        bool is_used = false;   // Has been used in a region.
        bool is_occupied = false;   // Has been searched when region is growing. It will be cleared later.
    };

    struct RegionParam {
        std::vector<PixelParam *> pixels;
        float angle = 0.0f;
    };

    struct RectangleParam {
        Vec2 start_point = Vec2::Zero();
        Vec2 end_point = Vec2::Zero();
        Vec2 center_point = Vec2::Zero();
        float length = 0.0f;
        float width = 0.0f;
        float angle = 0.0f;
        Vec2 dir_vector = Vec2::Identity();
        float inlier_ratio = 0.0f;
    };

    struct Options {
        float kMinValidGradientNorm = 20.0f;
        float kMinToleranceAngleResidualInRad = 22.5f * kDegToRad;
        float kMinValidLineLengthInPixel = 20.0f;
        float kMaxToleranceInlierRation = 0.6f;
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
    std::vector<PixelParam *> &sorted_pixels() { return sorted_pixels_; }
    std::vector<RectangleParam> &rectangles() { return rectangles_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }
    const Eigen::Matrix<PixelParam, Eigen::Dynamic, Eigen::Dynamic> &pixels() const { return pixels_; }
    const std::vector<PixelParam *> &sorted_pixels() const { return sorted_pixels_; }
    const std::vector<RectangleParam> &rectangles() const { return rectangles_; }

private:
    bool ComputeLineLevelAngleMap(const GrayImage &image);
    void GrowRegion(PixelParam &seed_pixel, RegionParam &region);
    void TryToAddPixelIntoCandidates(PixelParam &neighbour);
    RectangleParam ConvertRegionToRectangle(const RegionParam &region);

private:
    Options options_;

    Eigen::Matrix<PixelParam, Eigen::Dynamic, Eigen::Dynamic> pixels_;
    std::vector<PixelParam *> sorted_pixels_;
    CircularBuffer<PixelParam *, 1000> candidates_;
    CircularBuffer<PixelParam *, 1000> visited_pixels_;
    std::vector<RectangleParam> rectangles_;
};

}

#endif // end of _FEATURE_LINE_DETECTOR_H_
