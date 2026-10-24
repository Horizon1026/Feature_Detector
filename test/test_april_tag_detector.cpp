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

int main(int argc, char **argv) {
    ReportColorWarn(">> Test april tag detector");
    std::string image_file_path = "../examples/image.png";
    if (argc >= 2) {
        image_file_path = argv[1];
    }

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
