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

std::string image_file_path = "../examples/calibration_board_1.png";

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
    const auto &lines = detector.detected_lines();
    const auto &rectangles = detector.lines_idx_of_rect();
    for (const auto &rect: rectangles) {
        for (uint32_t i = 0; i < rect.size(); ++i) {
            const auto &line = lines[rect[i]];
            ImagePainter::DrawBressenhanLine(show_image, static_cast<int32_t>(line.start_point().x()), static_cast<int32_t>(line.start_point().y()),
                                             static_cast<int32_t>(line.end_point().x()), static_cast<int32_t>(line.end_point().y()), RgbColor::kGreen);
        }
    }

    Visualizor2D::ShowImage("Detected rectangles", show_image);
}

int main(int argc, char **argv) {
    ReportColorWarn(">> Test april tag detector");

    GrayImage image;
    Visualizor2D::LoadImage(image_file_path, image);

    AprilTagDetector detector;
    std::vector<AprilTag> april_tags;
    detector.DetectAprilTags(image, april_tags);

    DrawDetectedLinesOnImage(image, detector);
    DrawDetectedRetanglesOnImage(image, detector);
    Visualizor2D::WaitKey(0);
    return 0;
}
