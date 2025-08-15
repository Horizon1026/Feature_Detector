#include "basic_type.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "image_painter.h"
#include "visualizor_2d.h"
#include "tick_tock.h"

#include "nn_feature_point_detector.h"
#include "enable_stack_backward.h"

using namespace SLAM_UTILITY;
using namespace SLAM_VISUALIZOR;
using namespace FEATURE_DETECTOR;
using namespace IMAGE_PAINTER;

void ShowImage(const GrayImage &image, const std::string &title, const std::vector<Vec2> &features) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < features.size(); ++i) {
        ImagePainter::DrawSolidCircle(show_image, static_cast<int32_t>(features[i].x()), static_cast<int32_t>(features[i].y()), 4, RgbColor::kCyan);
    }
    Visualizor2D::ShowImage(title, show_image);
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test nn feature point detector." RESET_COLOR);

    // Load the image.
    GrayImage image;
    Visualizor2D::LoadImage("../examples/image.png", image);

    // Add some exists points. New feature points should not be detected around them.
    std::vector<Vec2> all_pixel_uv;
    for (int32_t i = 1; i < 5; ++i) {
        for (int32_t j = 1; j < 5; ++j) {
            all_pixel_uv.emplace_back(Vec2(i * 15, j * 15));
        }
    }

    // Initialize feature detector.
    NNFeaturePointDetector detector;
    detector.options().kMinResponse = 0.1f;
    detector.options().kMinFeatureDistance = 20;
    detector.options().kMaxNumberOfDetectedFeatures = 100;
    detector.options().kModelType = NNFeaturePointDetector::ModelType::kSuperpointNms;
    detector.options().kMaxImageRows = image.rows();
    detector.options().kMaxImageCols = image.cols();
    detector.Initialize();

    // Detect feature points.
    TickTock timer;
    std::vector<SuperpointDescriptorType> descriptors;
    detector.DetectGoodFeaturesWithDescriptor(image, all_pixel_uv, descriptors);
    ReportInfo("Superpoint detect time cost " << timer.TockTickInMillisecond() << " ms.");

    // Show detect result.
    ShowImage(image, "Superpoint detected features", all_pixel_uv);
    ReportInfo("Superpoint detected " << all_pixel_uv.size());
    Visualizor2D::WaitKey(0);
    return 0;
}
