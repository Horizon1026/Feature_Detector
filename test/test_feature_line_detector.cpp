#include "feature_line_detector.h"
#include "log_report.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "image_painter.h"
#include "visualizor.h"

using namespace FEATURE_DETECTOR;
using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;

std::string image_file_path = "../examples/image.png";

void ShowImage(const GrayImage &image, const std::string &title, const std::vector<Vec4> &features) {
    uint8_t *buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(buf, image.rows(), image.cols(), true);
    ImagePainter::ConvertUint8ToRgb(image.data(), show_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < features.size(); ++i) {
        ImagePainter::DrawBressenhanLine(show_image, static_cast<int32_t>(features[i][0]), static_cast<int32_t>(features[i][1]),
            static_cast<int32_t>(features[i][2]), static_cast<int32_t>(features[i][3]), RgbColor::kCyan);
    }
    Visualizor::ShowImage(title, show_image);
}

void TestLsdFeatureLineDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Harris Feature Detector." RESET_COLOR);

    FeatureLineDetector detector;

    TickTock timer;
    std::vector<Vec4> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);
    ReportDebug("LSD line detect time cost " << timer.TockTickInMillisecond() << " ms.");

    ShowImage(image, "LSD line detected features", features);
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
