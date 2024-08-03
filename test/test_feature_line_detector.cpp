#include "feature_line_detector.h"
#include "log_report.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "image_painter.h"
#include "visualizor.h"

using namespace FEATURE_DETECTOR;
using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;
using namespace SLAM_UTILITY;

std::string image_file_path = "../examples/image.png";

void ShowDetectResult(const GrayImage &image, const std::string &title, const std::vector<Vec4> &features) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < features.size(); ++i) {
        ImagePainter::DrawBressenhanLine(show_image, static_cast<int32_t>(features[i][0]), static_cast<int32_t>(features[i][1]),
            static_cast<int32_t>(features[i][2]), static_cast<int32_t>(features[i][3]), RgbColor::kCyan);
    }
    Visualizor::ShowImage(title, show_image);
}

void ShowPixelsGradientNorm(const FeatureLineDetector &detector, const std::string &title) {
    const auto &pixels = detector.pixels();
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(pixels.rows() * pixels.cols() * sizeof(uint8_t));
    GrayImage show_image(buf, pixels.rows(), pixels.cols(), true);
    for (uint32_t col = 0; col < pixels.cols(); ++col) {
        for (uint32_t row = 0; row < pixels.rows(); ++row) {
            const uint8_t pixel_value = static_cast<uint8_t>(pixels(row, col).gradient_norm);
            show_image.SetPixelValueNoCheck(row, col, pixel_value);
        }
    }
    Visualizor::ShowImage(title, show_image);
}

void ShowPixelsValidation(const FeatureLineDetector &detector, const std::string &title) {
    const auto &pixels = detector.pixels();
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(pixels.rows() * pixels.cols() * sizeof(uint8_t));
    GrayImage show_image(buf, pixels.rows(), pixels.cols(), true);
    for (uint32_t col = 0; col < pixels.cols(); ++col) {
        for (uint32_t row = 0; row < pixels.rows(); ++row) {
            show_image.SetPixelValueNoCheck(row, col, pixels(row, col).is_valid ? 0 : 255);
        }
    }
    Visualizor::ShowImage(title, show_image);
}

void ShowPixelsGradientAngle(const FeatureLineDetector &detector, const std::string &title) {
    const auto &pixels = detector.pixels();
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(pixels.rows() * pixels.cols() * sizeof(uint8_t));
    GrayImage show_image(buf, pixels.rows(), pixels.cols(), true);
    for (uint32_t col = 0; col < pixels.cols(); ++col) {
        for (uint32_t row = 0; row < pixels.rows(); ++row) {
            if (pixels(row, col).is_valid) {
                const uint8_t pixel_value = static_cast<uint8_t>((pixels(row, col).line_level_angle + kPai) / k2Pai * 255.0f);
                show_image.SetPixelValueNoCheck(row, col, pixel_value);
            } else {
                show_image.SetPixelValueNoCheck(row, col, 0);
            }
        }
    }
    Visualizor::ShowImage(title, show_image);
}

void ShowUsedPixels(const FeatureLineDetector &detector, const std::string &title) {
    const auto &pixels = detector.pixels();
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(pixels.rows() * pixels.cols() * sizeof(uint8_t));
    GrayImage show_image(buf, pixels.rows(), pixels.cols(), true);
    for (uint32_t col = 0; col < pixels.cols(); ++col) {
        for (uint32_t row = 0; row < pixels.rows(); ++row) {
            if (pixels(row, col).is_used) {
                show_image.SetPixelValueNoCheck(row, col, 255);
            } else {
                show_image.SetPixelValueNoCheck(row, col, 0);
            }
        }
    }
    Visualizor::ShowImage(title, show_image);
}

void ShowDetectedRectangles(const GrayImage &image, const std::string &title, const FeatureLineDetector &detector) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (const auto &rect : detector.rectangles()) {
        ImagePainter::DrawSolidCircle(show_image, rect.center_point.x(), rect.center_point.y(), 2, RgbColor::kRed);
        ImagePainter::DrawBressenhanLine(show_image, rect.start_point.x(), rect.start_point.y(), rect.end_point.x(), rect.end_point.y(), RgbColor::kBlue);
        const Vec2 dir_vector = rect.dir_vector * 10.0f;
        ImagePainter::DrawBressenhanLine(show_image, rect.center_point.x(), rect.center_point.y(), rect.center_point.x() + dir_vector.x(), rect.center_point.y() + dir_vector.y(), RgbColor::kGreen);

    }
    Visualizor::ShowImage(title, show_image);
}

void TestLsdFeatureLineDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Lsd Line Feature Detector." RESET_COLOR);

    FeatureLineDetector detector;

    TickTock timer;
    std::vector<Vec4> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);
    ReportDebug("LSD line detect time cost " << timer.TockTickInMillisecond() << " ms.");

    ShowPixelsGradientNorm(detector, "pixel gradient norm");
    ShowPixelsValidation(detector, "pixel is valid");
    ShowPixelsGradientAngle(detector, "pixel gradient direction");
    ShowUsedPixels(detector, "pixel is used after region grow");
    ShowDetectedRectangles(image, "detected rectangles", detector);
    ShowDetectResult(image, "LSD line detected features", features);
    ReportInfo("LSD line detected " << features.size());
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test feature line detector." RESET_COLOR);
    int32_t feature_num_need = 200;

    GrayImage image;
    Visualizor::LoadImage(image_file_path, image);

    TestLsdFeatureLineDetector(image, feature_num_need);

    Visualizor::WaitKey(0);

    return 0;
}
