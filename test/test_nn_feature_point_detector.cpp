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
    TickTock timer;

    // Load the image.
    GrayImage image;
    Visualizor2D::LoadImage("../examples/image.png", image);

    // Initialize the detector.
    timer.TockTickInSecond();
    NNFeaturePointDetector detector("../src/nn_feature_point_detector/models/xfeat_cpu_1_1_h_w.pt");
    detector.options().kModelType = NNFeaturePointDetector::ModelType::kXFeat;
    ReportInfo("Load model cost " << timer.TockTickInSecond() << " s.");

    // Detect feature points with descriptors.
    std::vector<Vec2> features;
    std::vector<XFeatDescriptorType> descriptors;
    timer.TockTickInSecond();
    detector.DetectGoodFeaturesWithDescriptor(image, 200, features, descriptors);
    ReportInfo("NN feature detect cost " << timer.TockTickInSecond() << " s.");

    // Show the image of heat map.
    MatImg heatmap_mat_img = (detector.keypoints_heat_map() * 255.0f).cast<uint8_t>();
    GrayImage heatmap_image(heatmap_mat_img.data(), heatmap_mat_img.rows(), heatmap_mat_img.cols(), false);
    Visualizor2D::ShowImage("Heat Map", heatmap_image);

    // Show the detected features.
    ShowImage(image, "nn detected features", features);
    Visualizor2D::WaitKey(0);

    return 0;
}
