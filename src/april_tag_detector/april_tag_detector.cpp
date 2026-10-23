#include "april_tag_detector.h"

#include "extend_kd_tree.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include "numeric"

namespace feature_detector {

void AprilTagDetector::Reset() {
    detected_lines_.clear();
    lines_idx_of_rect_.clear();
}

bool AprilTagDetector::DetectAprilTags(const GrayImage &image, std::vector<AprilTag> &april_tags) {
    RETURN_FALSE_IF(image.rows() == 0 || image.cols() == 0 || image.data() == nullptr);
    Reset();

    // Detect line segments in image.
    line_detector_.DetectGoodFeatures(image, 300, detected_lines_);

    // Find rectangles in lines.
    if (!TryToFindRectanglesInDetectedLines()) {
        ReportWarn("[AprilTagDetector] Failed to detect rectangles in lines.");
        return false;
    }

    // Find apriltags in rectangles.
    if (!TryToFindAprilTagsInDetectedRectangles()) {
        ReportWarn("[AprilTagDetector] Failed to detect apriltags in rectangles.");
        return false;
    }

    return true;
}

bool AprilTagDetector::TryToFindRectanglesInDetectedLines() {
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
    // Lambda accessor to get position from data unit.
    auto accessor = [&](int32_t index) -> Vec2 {
        return end_points[index].pixel_uv;
    };
    // Prepare indices.
    std::vector<int32_t> indices(end_points.size());
    std::iota(indices.begin(), indices.end(), 0);
    // Build extend kd-tree.
    using KdTreeType = ExtendKdTreeNode<float, 2>;
    std::unique_ptr<KdTreeType> kd_tree_ptr = std::make_unique<KdTreeType>();
    kd_tree_ptr->Construct(indices, accessor);

    // Search rectangles.
    struct LineIndexType {
        uint32_t id = 0;
        bool is_inverse = false;
    };
    std::vector<LineIndexType> lines_id_in_rect;
    lines_idx_of_rect_.clear();
    for (uint32_t line_id = 0; line_id < detected_lines_.size(); ++line_id) {
        CONTINUE_IF(is_used[line_id]);
        is_used[line_id] = true;
        if (lines_id_in_rect.empty()) {
            lines_id_in_rect.emplace_back(LineIndexType {.id = line_id, .is_inverse = false});
        }
        while (lines_id_in_rect.size() != 4) {
            const uint32_t prev_line_id = lines_id_in_rect.back().id;
            const Vec2 prev_line_end_points =
                lines_id_in_rect.back().is_inverse ? detected_lines_[prev_line_id].start_point() : detected_lines_[prev_line_id].end_point();

            // Try to find next line with nearby start_points.
            bool succeed_to_match = false;
            std::multimap<float, int32_t> result_of_radius;
            kd_tree_ptr->SearchRadius(prev_line_end_points, options_.kMaxTolerancePixelDistanceToLinkLineEndPoints, accessor, result_of_radius);
            for (const auto &item: result_of_radius) {
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

            // If being here, it means no more lines can be matched.
            for (auto &item: lines_id_in_rect) {
                is_used[item.id] = false;
            }
            break;
        }

        if (lines_id_in_rect.size() == 4) {
            // Check is head and tail matched.
            const Vec2 start_point =
                lines_id_in_rect[0].is_inverse ? detected_lines_[lines_id_in_rect[0].id].end_point() : detected_lines_[lines_id_in_rect[0].id].start_point();
            const Vec2 end_point =
                lines_id_in_rect[3].is_inverse ? detected_lines_[lines_id_in_rect[3].id].start_point() : detected_lines_[lines_id_in_rect[3].id].end_point();
            if ((start_point - end_point).norm() < options_.kMaxTolerancePixelDistanceToLinkLineEndPoints) {
                lines_idx_of_rect_.emplace_back(
                    std::array<uint32_t, 4> {lines_id_in_rect[0].id, lines_id_in_rect[1].id, lines_id_in_rect[2].id, lines_id_in_rect[3].id});
            } else {
                for (const auto &item: lines_id_in_rect) {
                    is_used[item.id] = false;
                }
            }
        } else {
            for (const auto &item: lines_id_in_rect) {
                is_used[item.id] = false;
            }
        }
        lines_id_in_rect.clear();
    }

    return true;
}

bool AprilTagDetector::TryToFindAprilTagsInDetectedRectangles() {
    RETURN_FALSE_IF(lines_idx_of_rect_.empty());

    return true;
}

}  // namespace feature_detector
