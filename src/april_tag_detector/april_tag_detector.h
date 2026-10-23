#ifndef _APRIL_TAG_DETECTOR_H_
#define _APRIL_TAG_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "feature_line_detector.h"
#include "feature_point_shi_tomas_detector.h"
#include "line_segment.h"

#include "array"

namespace feature_detector {

enum class AprilTagFamily : uint8_t {
    kTag41H12 = 0,
    kTag36H11 = 1,
    kUnknown = 2,
};

struct AprilTag {
    AprilTagFamily famliy;
    int32_t id = -1;
    std::array<Vec2, 4> vertices;
};

/* Class AprilTagDetector Declaration. */
class AprilTagDetector final {

public:
    struct Options {
        float kMaxTolerancePixelDistanceToLinkLineEndPoints = 4.0f;
    };

public:
    AprilTagDetector() = default;
    ~AprilTagDetector() = default;

    bool DetectAprilTags(const GrayImage &image, std::vector<AprilTag> &april_tags);

    // Reference for member variables.
    std::vector<LineSegment2D> &detected_lines() { return detected_lines_; }
    std::vector<std::array<uint32_t, 4>> &lines_idx_of_rect() { return lines_idx_of_rect_; }
    // Const reference for member variables.
    const std::vector<LineSegment2D> &detected_lines() const { return detected_lines_; }
    const std::vector<std::array<uint32_t, 4>> &lines_idx_of_rect() const { return lines_idx_of_rect_; }

private:
    void Reset();
    bool TryToFindRectanglesInDetectedLines();
    bool TryToFindAprilTagsInDetectedRectangles();

private:
    Options options_;
    FeatureLineDetector line_detector_;
    FeaturePointShiTomasDetector point_detector_;

    std::vector<LineSegment2D> detected_lines_;
    std::vector<std::array<uint32_t, 4>> lines_idx_of_rect_;
};

}  // namespace feature_detector

#endif  // end of _APRIL_TAG_DETECTOR_H_
