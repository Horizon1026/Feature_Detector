#include "log_report.h"
#include "feature_point_detector.h"
#include "feature_harris.h"
#include "feature_shi_tomas.h"
#include "feature_fast.h"

using namespace FEATURE_DETECTOR;

#include "opencv2/opencv.hpp"

std::string image_file_path = "../examples/image.png";

void TestHarrisFeatureDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Harris Feature Detector." RESET_COLOR);

    FeaturePointDetector<HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 30.0f;

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
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("harris detected new features", show_image);

    ReportInfo("harris detected " << features.size());

}

void TestShiTomasFeatureDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Shi-Tomas Feature Detector." RESET_COLOR);

    FeaturePointDetector<ShiTomasFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 40.0f;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("shi tomas detected features", show_image);

    ReportInfo("shi tomas detected " << features.size());

}

void TestFastFeatureDetector(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">> Test Fast Feature Detector." RESET_COLOR);

    FeaturePointDetector<FastFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 10.0f;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("fast detected features", show_image);

    ReportInfo("fast detected " << features.size());
}

void TestOpencvDetectGoodFeatures(GrayImage &image, int32_t feature_num_need) {
    ReportInfo(YELLOW ">>Test Opencv Detect Good Features." RESET_COLOR);
    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());

    std::vector<cv::Point2f> features;
    cv::goodFeaturesToTrack(cv_image, features, feature_num_need, 0.01, 20);

    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, features[i], 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("opencv detected features", show_image);
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test feature detector." RESET_COLOR);
    int32_t feature_num_need = 200;

    cv::Mat raw_image = cv::imread(image_file_path, 0);
    GrayImage image(raw_image.data, raw_image.rows, raw_image.cols);

    TestOpencvDetectGoodFeatures(image, feature_num_need);
    TestFastFeatureDetector(image, feature_num_need);
    TestHarrisFeatureDetector(image, feature_num_need);
    TestShiTomasFeatureDetector(image, feature_num_need);
    TestUpdateMaskWithDetectedFeatures(image, feature_num_need);

    cv::waitKey(0);

    return 0;
}
