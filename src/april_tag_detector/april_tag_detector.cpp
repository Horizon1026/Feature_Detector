#include "april_tag_detector.h"

#include "extend_kd_tree.h"
#include "image_processor.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include "algorithm"
#include "bit"
#include "numeric"
#include "set"

namespace feature_detector {

void AprilTagDetector::Reset() {
    detected_lines_.clear();
    detected_rectangles_.clear();
}

void AprilTagDetector::CorrectOptions() {
    const auto *family_info = AprilTagDefinitions::GetFamilyInfo(options_.kTargetFamily);
    if (family_info != nullptr && family_info->reversed_border) {
        options_.kBlackBoundaryWidth = 1;
    }
}

bool AprilTagDetector::DetectAprilTags(const GrayImage &image, std::vector<AprilTag> &april_tags) {
    RETURN_FALSE_IF(image.rows() == 0 || image.cols() == 0 || image.data() == nullptr);
    Reset();
    CorrectOptions();

    // Keep the input and boundary pixels intact; Process() writes only the interior.
    std::vector<uint8_t> blurred_buffer(image.data(), image.data() + static_cast<size_t>(image.rows()) * image.cols());
    GrayImage blurred_image(blurred_buffer.data(), image.rows(), image.cols());
    image_processor::ImageProcessor processor;
    processor.kernel() = (Mat5() << 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1).finished();
    processor.kernel() *= 25.0f / 256.0f;  // Process() also divides by the 5x5 kernel size.
    RETURN_FALSE_IF_FALSE(processor.Process(image, blurred_image));

    // Detect line segments in image.
    line_detector_.DetectGoodFeatures(blurred_image, 300, detected_lines_);

    // Find rectangles in lines.
    if (!TryToFindRectanglesInDetectedLines(blurred_image)) {
        ReportWarn("[AprilTagDetector] Failed to detect rectangles in lines.");
        return false;
    }

    // Find apriltags in rectangles.
    if (!TryToFindAprilTagsInDetectedRectangles(image, april_tags)) {
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

bool AprilTagDetector::TryToFindAprilTagsInDetectedRectangles(const GrayImage &image, std::vector<AprilTag> &april_tags) {
    RETURN_FALSE_IF(detected_rectangles_.empty());
    april_tags.clear();
    april_tags.reserve(detected_rectangles_.size());

    const AprilTagDefinitions::FamilyInfo *family_info = AprilTagDefinitions::GetFamilyInfo(options_.kTargetFamily);
    RETURN_FALSE_IF(family_info == nullptr);
    const int32_t min_valid_hamming_distance = std::min(options_.kMaxToleranceHammingDistance, (family_info->min_valid_hamming_distance - 1) / 2);
    for (const auto &rect_vertices: detected_rectangles_) {
        int32_t best_april_tag_id = -1;
        int32_t min_hamming_distance = 100;
        int32_t best_direction = 0;
        std::vector<Vec2> sample_points;
        std::vector<Vec2> best_sample_points;

        // Try to decode april tag in different direction.
        for (uint8_t dir = 0; dir < static_cast<uint8_t>(AprilTagSampleDirection::kMaxDirectionNumber); ++dir) {
            AprilTagSampleDirection direction = static_cast<AprilTagSampleDirection>(dir);
            sample_points.clear();
            sample_points = GetSamplePointOfAprilTag(options_.kTargetFamily, direction, rect_vertices);
            CONTINUE_IF(sample_points.empty());
            const uint64_t code = DecodeAprilTag(image, sample_points);
            CONTINUE_IF(code == 0);

            // Try to find cloest tag.
            for (uint32_t id = 0; id < family_info->codes->size(); ++id) {
                const auto &family_code = (*family_info->codes)[id];
                const int32_t distance = HammingDistance(code, family_code);
                CONTINUE_IF(distance > min_valid_hamming_distance);
                if (distance < min_hamming_distance) {
                    min_hamming_distance = distance;
                    best_april_tag_id = id;
                    best_sample_points = sample_points;
                    best_direction = static_cast<int32_t>(direction);
                }
                BREAK_IF(min_hamming_distance == 0);
            }
        }

        // Validate apriltag.
        CONTINUE_IF(best_april_tag_id < 0);
        CONTINUE_IF(min_hamming_distance > min_valid_hamming_distance);
        april_tags.emplace_back(AprilTag {
            .famliy = options_.kTargetFamily,
            .id = best_april_tag_id,
            .vertices = rect_vertices,
            .sample_points = best_sample_points,
            .hamming_distance = min_hamming_distance,
        });
        for (int32_t i = 0; i < 4; ++i) {
            april_tags.back().vertices[i] = rect_vertices[(i + best_direction) % 4];
        }
    }
    return true;
}

std::vector<Vec2> AprilTagDetector::GetSamplePointOfAprilTag(const AprilTagFamily &famliy, const AprilTagSampleDirection &direction,
                                                             const std::array<Vec2, 4> &vertices) {
    const auto *family_info = AprilTagDefinitions::GetFamilyInfo(famliy);
    if (family_info == nullptr) {
        return {};
    }
    const auto *bits = family_info->bits;
    const uint32_t bit_num = family_info->bits->size();
    if (options_.kBlackBoundaryWidth < 1) {
        return {};
    }
    const float bit_offset = static_cast<float>(options_.kBlackBoundaryWidth - 1);
    const float border_width = family_info->border_width + 2.0f * bit_offset;
    switch (direction) {
        case AprilTagSampleDirection::kLeft:
        case AprilTagSampleDirection::kUp:
        case AprilTagSampleDirection::kRight:
        case AprilTagSampleDirection::kDown:
            break;
        default:
            return {};
    }

    // Reject non-convex/degenerate quads, then use clockwise image winding.
    float winding = 0.0f;
    for (uint32_t i = 0; i < 4; ++i) {
        if (!vertices[i].allFinite()) {
            return {};
        }
        const Vec2 edge1 = vertices[(i + 1) % 4] - vertices[i];
        const Vec2 edge2 = vertices[(i + 2) % 4] - vertices[(i + 1) % 4];
        const float cross = edge1.x() * edge2.y() - edge1.y() * edge2.x();
        if (!std::isfinite(cross) || cross == 0.0f || (i > 0 && cross * winding <= 0.0f)) {
            return {};
        }
        winding = cross;
    }
    auto corners = vertices;
    if (winding < 0.0f) {
        std::swap(corners[1], corners[3]);
    }

    // Fit a projective transform from the unit square to the detected border.
    static constexpr float unit_corners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Mat8 matrix_A = Mat8::Zero();
    Vec8 vector_b = Vec8::Zero();
    for (uint32_t i = 0; i < 4; ++i) {
        const float u = unit_corners[i][0];
        const float v = unit_corners[i][1];
        const float x = corners[i].x();
        const float y = corners[i].y();
        matrix_A.row(2 * i) << u, v, 1, 0, 0, 0, -x * u, -x * v;
        matrix_A.row(2 * i + 1) << 0, 0, 0, u, v, 1, -y * u, -y * v;
        vector_b(2 * i) = x;
        vector_b(2 * i + 1) = y;
    }
    const Eigen::ColPivHouseholderQR<Mat8> solver(matrix_A);
    if (solver.rank() != 8) {
        return {};
    }
    const Vec8 h = solver.solve(vector_b);
    if (!h.allFinite()) {
        return {};
    }

    std::vector<Vec2> sample_points;
    sample_points.reserve(bit_num);
    for (uint32_t i = 0; i < bit_num; ++i) {
        // Standard41h12 has data outside its border; do not clamp to [0, 1].
        const float u = ((*bits)[i][0] + bit_offset + 0.5f) / border_width;
        const float v = ((*bits)[i][1] + bit_offset + 0.5f) / border_width;
        float rotated_u = u;
        float rotated_v = v;
        switch (direction) {
            case AprilTagSampleDirection::kLeft:
                rotated_u = v;
                rotated_v = 1.0 - u;
                break;
            case AprilTagSampleDirection::kRight:
                rotated_u = 1.0 - v;
                rotated_v = u;
                break;
            case AprilTagSampleDirection::kDown:
                rotated_u = 1.0 - u;
                rotated_v = 1.0 - v;
                break;
            default:
                break;
        }
        const float denominator = h(6) * rotated_u + h(7) * rotated_v + 1.0f;
        if (!std::isfinite(denominator) || denominator <= kZeroFloat) {
            return {};
        }
        const Vec2 point((h(0) * rotated_u + h(1) * rotated_v + h(2)) / denominator, (h(3) * rotated_u + h(4) * rotated_v + h(5)) / denominator);
        if (!point.allFinite()) {
            return {};
        }
        sample_points.emplace_back(point);
    }
    return sample_points;
}

uint64_t AprilTagDetector::DecodeAprilTag(const GrayImage &image, const std::vector<Vec2> &sample_points) {
    uint64_t code = 0;
    float value = 0;

    // Dynamic threshold.
    float max_value = 0.0f;
    float min_value = 255.0f;
    for (const Vec2 &point: sample_points) {
        if (image.GetPixelValue(point.y(), point.x(), &value)) {
            max_value = std::max(max_value, value);
            min_value = std::min(min_value, value);
        }
    }
    const float threshold = 0.5f * (min_value + max_value);
    if (max_value - min_value < options_.kMinTolerancePixelValueDiffBetweenWhiteAndBlack) {
        return code;
    }

    // Decode bits.
    for (const Vec2 &point: sample_points) {
        code <<= 1;
        if (image.GetPixelValue(point.y(), point.x(), &value)) {
            if (value > threshold) {
                ++code;
            }
        }
    }
    return code;
}

int32_t AprilTagDetector::HammingDistance(const uint64_t code_0, const uint64_t code_1) { return std::popcount(code_0 ^ code_1); }

}  // namespace feature_detector
