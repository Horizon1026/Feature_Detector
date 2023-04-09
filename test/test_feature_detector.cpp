#include "log_api.h"
#include "feature_detector.h"

#include "opencv2/opencv.hpp"

std::string image_file_path = "../examples/image.png";

void test_harris_detector(Image &image, int32_t feature_num_need) {
    LogInfo("test_harris_detector.");

    FEATURE_DETECTOR::FeatureDetector detector;
    detector.options().kMethod = FEATURE_DETECTOR::FeatureDetector::HARRIS;
    detector.options().kMinValidResponse = 20.0f;
    detector.options().kMinFeatureDistance = 20;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("harris detected features", show_image);

    LogInfo("harris detected " << features.size());

}

void test_harris_detector_with_detected_features(Image &image, int32_t feature_num_need) {
    LogInfo("test_harris_detector, but some features has been detected.");

    FEATURE_DETECTOR::FeatureDetector detector;
    detector.options().kMethod = FEATURE_DETECTOR::FeatureDetector::HARRIS;
    detector.options().kMinValidResponse = 20.0f;
    detector.options().kMinFeatureDistance = 20;

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

    LogInfo("harris detected " << features.size());

}

void test_shi_tomas_detector(Image &image, int32_t feature_num_need) {
    LogInfo("test_shi_tomas_detector.");

    FEATURE_DETECTOR::FeatureDetector detector;
    detector.options().kMethod = FEATURE_DETECTOR::FeatureDetector::SHI_TOMAS;
    detector.options().kMinValidResponse = 40.0f;
    detector.options().kMinFeatureDistance = 20;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("shi tomas detected features", show_image);

    LogInfo("shi tomas detected " << features.size());

}

void test_fast_detector(Image &image, int32_t feature_num_need) {
    LogInfo("test_fast_detector.");

    FEATURE_DETECTOR::FeatureDetector detector;
    detector.options().kMethod = FEATURE_DETECTOR::FeatureDetector::FAST;
    detector.options().kMinValidResponse = 10.0f;
    detector.options().kMinFeatureDistance = 20;

    std::vector<Vec2> features;
    detector.DetectGoodFeatures(image, feature_num_need, features);

    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < features.size(); i++) {
        cv::circle(show_image, cv::Point2f(features[i].x(), features[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("fast detected features", show_image);

    LogInfo("fast detected " << features.size());
}

void test_cv_good_feature(Image &image, int32_t feature_num_need) {
    LogInfo("test_cv_good_feature.");
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

int main() {
    LogInfo("Test feature detector.");
    int32_t feature_num_need = 200;

    cv::Mat raw_image = cv::imread(image_file_path, 0);
    Image image(raw_image.data, raw_image.rows, raw_image.cols);

    test_cv_good_feature(image, feature_num_need);
    test_fast_detector(image, feature_num_need);
    test_harris_detector(image, feature_num_need);
    test_shi_tomas_detector(image, feature_num_need);
    test_harris_detector_with_detected_features(image, feature_num_need);

    cv::waitKey(0);

    return 0;
}