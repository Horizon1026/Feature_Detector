#include "feature_point_detector.h"
#include "feature_harris.h"
#include "descriptor_brief.h"

#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "image_painter.h"
#include "visualizor_2d.h"

using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;

std::string image_file_path = "../examples/image.png";

std::vector<Vec2> TestHarrisFeatureDetector(const GrayImage &image, const int32_t feature_num_need) {
    ReportInfo(">> Test Harris Feature Detector.");

    FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 20.0f;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < features.size(); i++) {
        ImagePainter::DrawSolidCircle(show_image, static_cast<int32_t>(features[i].x()), static_cast<int32_t>(features[i].y()), 4, RgbColor::kRed);
    }
    Visualizor2D::ShowImage("harris detected features", show_image);

    ReportInfo("harris detected " << features.size());
    return features;
}

void TestBriefDescriptor(const GrayImage &image, const std::vector<Vec2> &features) {
    ReportInfo(">> Test Brief Feature Descriptor.");

    FEATURE_DETECTOR::BriefDescriptor descriptor;
    descriptor.options().kHalfPatchSize = 8;
    descriptor.options().kLength = 128;
    descriptor.options().kValidBoundary = 16;

    TickTock timer;
    std::vector<FEATURE_DETECTOR::BriefType> descriptors;
    descriptor.Compute(image, features, descriptors);
    ReportDebug("Compute descriptor cost time " << timer.TockTickInMillisecond() << " ms.");

    for (const auto &item: descriptors) {
        ReportText("descriptor is ");
        for (const auto &bit: item) {
            ReportText(static_cast<int32_t>(bit));
        }
        ReportText(std::endl);
    }
}

int main(int argc, char **argv) {
    ReportInfo("Test feature detector.");
    int32_t feature_num_need = 10;

    GrayImage image;
    Visualizor2D::LoadImage(image_file_path, image);

    std::vector<Vec2> features = TestHarrisFeatureDetector(image, feature_num_need);
    TestBriefDescriptor(image, features);

    Visualizor2D::WaitKey(0);

    return 0;
}
