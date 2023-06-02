#include "log_report.h"
#include "feature_point_detector.h"
#include "feature_harris.h"
#include "descriptor_brief.h"

#include "opencv2/opencv.hpp"

std::string image_file_path = "../examples/image.png";

std::vector<Vec2> TestHarrisFeatureDetector(const Image &image, const int32_t feature_num_need) {
    ReportInfo(">> Test Harris Feature Detector.");

    FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 20.0f;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("harris detected features", show_image);

    ReportInfo("harris detected " << features.size());
    return features;
}

void TestBriefDescriptor(const Image &image, const std::vector<Vec2> &features) {
    ReportInfo(">> Test Brief Feature Descriptor.");

    FEATURE_DETECTOR::BriefDescriptor descriptor;
    descriptor.options().kHalfPatchSize = 8;
    descriptor.options().kLength = 128;
    descriptor.options().kValidBoundary = 16;

    std::vector<FEATURE_DETECTOR::BriefType> descriptors;
    descriptor.Compute(image, features, descriptors);

    for (const auto &item : descriptors) {
        std::cout << "descriptor is ";
        for (const auto &bit : item) {
            std::cout << static_cast<int32_t>(bit);
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    ReportInfo("Test feature detector.");
    int32_t feature_num_need = 10;

    cv::Mat raw_image = cv::imread(image_file_path, 0);
    Image image(raw_image.data, raw_image.rows, raw_image.cols);

    std::vector<Vec2> features = TestHarrisFeatureDetector(image, feature_num_need);
    TestBriefDescriptor(image, features);

    cv::waitKey(0);

    return 0;
}
