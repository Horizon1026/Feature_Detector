#ifndef _APRIL_TAG_DETECTOR_H_
#define _APRIL_TAG_DETECTOR_H_

#include "april_tag_typedef.h"
#include "basic_type.h"
#include "datatype_image.h"
#include "feature_line_detector.h"
#include "feature_point_shi_tomas_detector.h"
#include "line_segment.h"

#include "array"

namespace feature_detector {

/* Class AprilTagDetector Declaration. */
class AprilTagDetector final {

public:
    struct Options {
        AprilTagFamily kTargetFamily = AprilTagFamily::kTag36H11;
        float kMaxTolerancePixelDistanceToLinkLineEndPoints = 4.0f;
        int32_t kMaxToleranceHammingDistance = 2;
        float kMinTolerancePixelValueDiffBetweenWhiteAndBlack = 127.0f;
        int32_t kBlackBoundaryWidth = 1;  // Black border thickness in cells; reversed-border families use 1.
    };

public:
    AprilTagDetector() = default;
    ~AprilTagDetector() = default;

    bool DetectAprilTags(const GrayImage &image, std::vector<AprilTag> &april_tags);

    // Reference for member variables.
    Options &options() { return options_; }
    FeatureLineDetector &line_detector() { return line_detector_; }
    FeaturePointShiTomasDetector &point_detector() { return point_detector_; }
    std::vector<LineSegment2D> &detected_lines() { return detected_lines_; }
    std::vector<std::array<Vec2, 4>> &detected_rectangles() { return detected_rectangles_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }
    const FeatureLineDetector &line_detector() const { return line_detector_; }
    const FeaturePointShiTomasDetector &point_detector() const { return point_detector_; }
    const std::vector<LineSegment2D> &detected_lines() const { return detected_lines_; }
    const std::vector<std::array<Vec2, 4>> &detected_rectangles() const { return detected_rectangles_; }

private:
    void Reset();
    void CorrectOptions();
    bool TryToFindRectanglesInDetectedLines(const GrayImage &image);
    bool TryToFindAprilTagsInDetectedRectangles(const GrayImage &image, std::vector<AprilTag> &april_tags);
    std::vector<Vec2> GetSamplePointOfAprilTag(const AprilTagFamily &famliy, const AprilTagSampleDirection &direction, const std::array<Vec2, 4> &vertices);
    uint64_t DecodeAprilTag(const GrayImage &image, const std::vector<Vec2> &sample_points);
    int32_t HammingDistance(const uint64_t code_0, const uint64_t code_1);

private:
    Options options_;
    FeatureLineDetector line_detector_;
    FeaturePointShiTomasDetector point_detector_;

    std::vector<LineSegment2D> detected_lines_;
    std::vector<std::array<Vec2, 4>> detected_rectangles_;
};

}  // namespace feature_detector

#endif  // end of _APRIL_TAG_DETECTOR_H_
