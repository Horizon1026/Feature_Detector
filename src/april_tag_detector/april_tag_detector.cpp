#include "april_tag_detector.h"

#include "extend_kd_tree.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include "algorithm"
#include "numeric"
#include "set"

namespace feature_detector {

void AprilTagDetector::Reset() {
    detected_lines_.clear();
    detected_rectangles_.clear();
}

bool AprilTagDetector::DetectAprilTags(const GrayImage &image, std::vector<AprilTag> &april_tags) {
    RETURN_FALSE_IF(image.rows() == 0 || image.cols() == 0 || image.data() == nullptr);
    Reset();

    // Detect line segments in image.
    line_detector_.DetectGoodFeatures(image, 300, detected_lines_);

    // Find rectangles in lines.
    if (!TryToFindRectanglesInDetectedLines(image)) {
        ReportWarn("[AprilTagDetector] Failed to detect rectangles in lines.");
        return false;
    }

    // Find apriltags in rectangles.
    if (!TryToFindAprilTagsInDetectedRectangles(image)) {
        ReportWarn("[AprilTagDetector] Failed to detect apriltags in rectangles.");
        return false;
    }

    return true;
}

bool AprilTagDetector::TryToFindRectanglesInDetectedLines(const GrayImage &image) {
    RETURN_FALSE_IF(detected_lines_.size() < 4);
    std::vector<bool> is_used;
    is_used.resize(detected_lines_.size(), false);

    // Extract all end points of detected lines.
    struct PointType {
        Vec2 pixel_uv = Vec2::Zero();
        uint32_t line_id = -1;
    };
    std::vector<PointType> end_points;
    end_points.reserve(detected_lines_.size() * 2);
    for (uint32_t i = 0; i < detected_lines_.size(); ++i) {
        const auto &line = detected_lines_[i];
        end_points.emplace_back(PointType {.pixel_uv = line.start_point(), .line_id = i});
        end_points.emplace_back(PointType {.pixel_uv = line.end_point(), .line_id = i});
    }
    // Lambda KdtreeAccessor to get position from data unit.
    auto KdtreeAccessor = [&](int32_t index) -> Vec2 {
        return end_points[index].pixel_uv;
    };
    // Prepare indices.
    std::vector<int32_t> indices(end_points.size());
    std::iota(indices.begin(), indices.end(), 0);
    // Build extend kd-tree.
    using KdTreeType = ExtendKdTreeNode<float, 2>;
    std::unique_ptr<KdTreeType> kd_tree_ptr = std::make_unique<KdTreeType>();
    kd_tree_ptr->Construct(indices, KdtreeAccessor);

    // Search rectangles.
    struct LineIndexType {
        uint32_t id = 0;
        bool is_inverse = false;
        uint32_t next_candidate = 0;
    };
    auto GetEndPoint = [&](LineIndexType line_index, bool need_start_point) -> Vec2 {
        if (line_index.is_inverse == need_start_point) {
            return detected_lines_[line_index.id].end_point();
        } else {
            return detected_lines_[line_index.id].start_point();
        }
    };
    std::vector<LineIndexType> lines_id_in_rect;
    std::set<std::array<uint32_t, 4>> unique_rectangles;
    detected_rectangles_.clear();
    for (uint32_t line_id = 0; line_id < detected_lines_.size(); ++line_id) {
        CONTINUE_IF(is_used[line_id]);
        is_used[line_id] = true;
        if (lines_id_in_rect.empty()) {
            lines_id_in_rect.emplace_back(LineIndexType {.id = line_id, .is_inverse = false});
        }
        while (!lines_id_in_rect.empty()) {
            if (lines_id_in_rect.size() == 4) {
                const Vec2 start_point = GetEndPoint(lines_id_in_rect[0], true);
                const Vec2 end_point = GetEndPoint(lines_id_in_rect[3], false);
                if ((start_point - end_point).norm() < options_.kMaxTolerancePixelDistanceToLinkLineEndPoints) {
                    const std::array<uint32_t, 4> rect = {lines_id_in_rect[0].id, lines_id_in_rect[1].id, lines_id_in_rect[2].id, lines_id_in_rect[3].id};
                    auto sorted_rect = rect;
                    std::sort(sorted_rect.begin(), sorted_rect.end());
                    if (unique_rectangles.insert(sorted_rect).second) {
                        detected_rectangles_.emplace_back(std::array<Vec2, 4> {
                            0.5f * (GetEndPoint(lines_id_in_rect[0], true) + GetEndPoint(lines_id_in_rect[3], false)),
                            0.5f * (GetEndPoint(lines_id_in_rect[1], true) + GetEndPoint(lines_id_in_rect[0], false)),
                            0.5f * (GetEndPoint(lines_id_in_rect[2], true) + GetEndPoint(lines_id_in_rect[1], false)),
                            0.5f * (GetEndPoint(lines_id_in_rect[3], true) + GetEndPoint(lines_id_in_rect[2], false)),
                        });
                    }
                }
                // Release this edge and continue searching other combinations.
                is_used[lines_id_in_rect.back().id] = false;
                lines_id_in_rect.pop_back();
                continue;
            }
            const uint32_t prev_line_id = lines_id_in_rect.back().id;
            const Vec2 prev_line_end_points =
                lines_id_in_rect.back().is_inverse ? detected_lines_[prev_line_id].start_point() : detected_lines_[prev_line_id].end_point();

            // Try to find next line with nearby start_points.
            bool succeed_to_match = false;
            std::multimap<float, int32_t> result_of_radius;
            kd_tree_ptr->SearchRadius(prev_line_end_points, options_.kMaxTolerancePixelDistanceToLinkLineEndPoints, KdtreeAccessor, result_of_radius);
            uint32_t candidate_index = 0;
            for (const auto &item: result_of_radius) {
                const uint32_t current_candidate_index = candidate_index;
                ++candidate_index;
                CONTINUE_IF(current_candidate_index < lines_id_in_rect.back().next_candidate);

                lines_id_in_rect.back().next_candidate = candidate_index;
                const uint32_t matched_line_id = item.second / 2;
                const bool is_start_point_matched = item.second % 2 == 0;
                CONTINUE_IF(matched_line_id == prev_line_id);
                CONTINUE_IF(is_used[matched_line_id]);

                lines_id_in_rect.emplace_back(LineIndexType {
                    .id = matched_line_id,
                    .is_inverse = !is_start_point_matched,
                });
                is_used[matched_line_id] = true;
                succeed_to_match = true;
                break;
            }
            CONTINUE_IF(succeed_to_match);

            // Backtrack and try the parent's next candidate.
            is_used[lines_id_in_rect.back().id] = false;
            lines_id_in_rect.pop_back();
        }

        lines_id_in_rect.clear();
    }

    // Refine vertices of each rectangle.
    for (auto &rectangle: detected_rectangles_) {
        for (auto &point: rectangle) {
            Vec2 refined_point = point;
            CONTINUE_IF(!point_detector_.RefineFeatureToSubPixelPrecision(image, point, refined_point));
            point = refined_point;
        }
    }

    return true;
}

bool AprilTagDetector::TryToFindAprilTagsInDetectedRectangles(const GrayImage &image) {
    RETURN_FALSE_IF(detected_rectangles_.empty());

    return true;
}

}  // namespace feature_detector
