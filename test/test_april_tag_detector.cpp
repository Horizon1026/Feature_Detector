#include "april_tag_detector.h"
#include "image_painter.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "slam_operations.h"
#include "visualizor_2d.h"

using namespace slam_utility;
using namespace feature_detector;
using namespace slam_visualizor;
using namespace image_painter;

void DrawDetectedLinesOnImage(const GrayImage &image, const AprilTagDetector &detector) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    const auto &lines = detector.detected_lines();
    for (uint32_t i = 0; i < lines.size(); ++i) {
        ImagePainter::DrawBressenhanLine(show_image, static_cast<int32_t>(lines[i].start_point().x()), static_cast<int32_t>(lines[i].start_point().y()),
                                         static_cast<int32_t>(lines[i].end_point().x()), static_cast<int32_t>(lines[i].end_point().y()), RgbColor::kRed);
    }
    Visualizor2D::ShowImage("Detected lines", show_image);
}

void DrawDetectedRetanglesOnImage(const GrayImage &image, const AprilTagDetector &detector) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    const auto &rectangles = detector.detected_rectangles();
    for (const auto &rect: rectangles) {
        for (uint32_t i = 0; i < rect.size(); ++i) {
            ImagePainter::DrawBressenhanLine(show_image, rect[0].x(), rect[0].y(), rect[1].x(), rect[1].y(), RgbColor::kRed);
            ImagePainter::DrawBressenhanLine(show_image, rect[1].x(), rect[1].y(), rect[2].x(), rect[2].y(), RgbColor::kRed);
            ImagePainter::DrawBressenhanLine(show_image, rect[2].x(), rect[2].y(), rect[3].x(), rect[3].y(), RgbColor::kRed);
            ImagePainter::DrawBressenhanLine(show_image, rect[3].x(), rect[3].y(), rect[0].x(), rect[0].y(), RgbColor::kRed);
        }
    }

    Visualizor2D::ShowImage("Detected rectangles", show_image);
}

void DrawDetectedAprilTagsOnImage(const GrayImage &image, const std::vector<AprilTag> &april_tags) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (const auto &tag: april_tags) {
        // Draw range of april tag.
        for (uint32_t i = 0; i < tag.vertices.size(); ++i) {
            ImagePainter::DrawBressenhanLine(show_image, tag.vertices[0].x(), tag.vertices[0].y(), tag.vertices[1].x(), tag.vertices[1].y(), RgbColor::kRed);
            ImagePainter::DrawBressenhanLine(show_image, tag.vertices[1].x(), tag.vertices[1].y(), tag.vertices[2].x(), tag.vertices[2].y(), RgbColor::kRed);
            ImagePainter::DrawBressenhanLine(show_image, tag.vertices[2].x(), tag.vertices[2].y(), tag.vertices[3].x(), tag.vertices[3].y(), RgbColor::kRed);
            ImagePainter::DrawBressenhanLine(show_image, tag.vertices[3].x(), tag.vertices[3].y(), tag.vertices[0].x(), tag.vertices[0].y(), RgbColor::kRed);
            // Draw first vertex of april tag.
            ImagePainter::DrawSolidCircle(show_image, tag.vertices[0].x(), tag.vertices[0].y(), 3.0f, RgbColor::kCyan);
        }

        // Draw april tag id.
        const Vec2 mid_point = 0.25f * (tag.vertices[0] + tag.vertices[1] + tag.vertices[2] + tag.vertices[3]);
        ImagePainter::DrawString(show_image, std::to_string(tag.id), mid_point.x(), mid_point.y(), RgbColor::kRed, 16);

        // Draw sample points.
        for (const Vec2 &sample: tag.sample_points) {
            ImagePainter::DrawSolidCircle(show_image, sample.x(), sample.y(), 1.0f, RgbColor::kOrange);
        }
    }

    Visualizor2D::ShowImage("Detected april tags", show_image);
    ReportInfo(april_tags.size() << " april tags detected.");
}

int main(int argc, char **argv) {
    ReportColorWarn(">> Test april tag detector.");
    std::string image_file_path = "../examples/image.png";
    if (argc >= 2) {
        image_file_path = argv[1];
    }
    std::string family_name = "tag_36h11";
    if (argc >= 3) {
        family_name = argv[2];
    }
    int32_t black_boundary_width = 1;
    if (argc >= 4) {
        black_boundary_width = std::stoi(argv[3]);
    }

    GrayImage image;
    Visualizor2D::LoadImage(image_file_path, image);

    AprilTagDetector detector;
    detector.options().kTargetFamily = AprilTagDefinitions::GetFamilyType(family_name);
    detector.options().kBlackBoundaryWidth = black_boundary_width;
    ReportInfo("Tag family id: " << static_cast<int32_t>(detector.options().kTargetFamily) << ", black boundary width is "
                                 << detector.options().kBlackBoundaryWidth);
    if (detector.options().kTargetFamily == AprilTagFamily::kUnknown) {
        ReportError("Unknown family type.");
        return 0;
    }

    std::vector<AprilTag> april_tags;
    detector.DetectAprilTags(image, april_tags);

    DrawDetectedLinesOnImage(image, detector);
    DrawDetectedRetanglesOnImage(image, detector);
    DrawDetectedAprilTagsOnImage(image, april_tags);
    Visualizor2D::WaitKey(0);
    return 0;
}
