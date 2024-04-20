#include "feature_point_detector.h"
#include "feature_harris.h"
#include "feature_shi_tomas.h"
#include "feature_fast.h"

#include "log_report.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "image_painter.h"
#include "visualizor.h"

using namespace FEATURE_DETECTOR;
using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;

std::string image_file_path = "../examples/image.png";

void ShowImage(const GrayImage &image, const std::string &title, const std::vector<Vec2> &features) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < features.size(); ++i) {
        ImagePainter::DrawSolidCircle(show_image, static_cast<int32_t>(features[i].x()), static_cast<int32_t>(features[i].y()), 4, RgbColor::kCyan);
    }
    Visualizor::ShowImage(title, show_image);
}

void TestHarrisFeatureDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Harris Feature Detector." RESET_COLOR);

    FeaturePointDetector<HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 30.0f;

    TickTock timer;
    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);
    ReportDebug("harris detect time cost " << timer.TockTickInMillisecond() << " ms.");

    ShowImage(image, "harris detected features", features);
    ReportInfo("harris detected " << features.size());
}

void TestUpdateMaskWithDetectedFeatures(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Harris Feature Detector, but some features has been detected." RESET_COLOR);

    FeaturePointDetector<HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 30.0f;

    std::vector<Vec2> features;
    features.reserve(feature_num_need);
    for (int32_t i = 1; i < 10; ++i) {
        for (int32_t j = 1; j < 10; ++j) {
            features.emplace_back(Vec2(i * 15, j * 15));
        }
    }

    TickTock timer;
    detector.DetectGoodFeatures(image, feature_num_need, features);
    ReportDebug("harris detect new features time cost " << timer.TockTickInMillisecond() << " ms.");

    ShowImage(image, "harris detected new features", features);
    ReportInfo("harris detected " << features.size());
}

void TestShiTomasFeatureDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Shi-Tomas Feature Detector." RESET_COLOR);

    FeaturePointDetector<ShiTomasFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 40.0f;

    TickTock timer;
    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);
    ReportDebug("shi tomas detect new features time cost " << timer.TockTickInMillisecond() << " ms.");

    ShowImage(image, "shi tomas detected features", features);
    ReportInfo("shi tomas detected " << features.size());
}

void TestFastFeatureDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Fast Feature Detector." RESET_COLOR);

    FeaturePointDetector<FastFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 10.0f;

    TickTock timer;
    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);
    ReportDebug("fast detect new features time cost " << timer.TockTickInMillisecond() << " ms.");

    ShowImage(image, "fast detected features", features);
    ReportInfo("fast detected " << features.size());
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test feature detector." RESET_COLOR);
    int32_t feature_num_need = 200;

    GrayImage image;
    Visualizor::LoadImage(image_file_path, image);

    TestFastFeatureDetector(image, feature_num_need);
    TestHarrisFeatureDetector(image, feature_num_need);
    TestShiTomasFeatureDetector(image, feature_num_need);
    TestUpdateMaskWithDetectedFeatures(image, feature_num_need);

    Visualizor::WaitKey(0);

    return 0;
}
