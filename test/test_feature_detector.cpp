#include "log_api.h"
#include "feature_detector.h"

#include "opencv2/opencv.hpp"

std::string image_file_path = "../examples/image.png";

void test_harris_detector(Image *image, int32_t feature_num_need) {
    LogInfo(GREEN ">> test_harris_detector." RESET_COLOR);

}

void test_fast_detector(Image *image, int32_t feature_num_need) {
    LogInfo(GREEN ">> test_fast_detector." RESET_COLOR);

}

void test_cv_good_feature(Image *image, int32_t feature_num_need) {
    LogInfo(GREEN ">> test_cv_good_feature." RESET_COLOR);

    cv::Mat cv_image(image->rows(), image->cols(), CV_8UC1, image->image_data());

    std::vector<cv::Point2f> ref_corners;
    cv::goodFeaturesToTrack(cv_image, ref_corners, feature_num_need, 0.01, 20);

    cv::Mat show_image(cv_image.rows, cv_image.cols, CV_8UC3);
    cv::cvtColor(cv_image, show_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < ref_corners.size(); i++) {
        cv::circle(show_image, ref_corners[i], 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("LK : Feature before multi tracking", show_image);
}

int main() {
    LogInfo(YELLOW ">> Test feature detector." RESET_COLOR);
    int32_t feature_num_need = 200;

    cv::Mat raw_image = cv::imread(image_file_path, 0);
    cv::imshow("raw_image", raw_image);

    Image image(raw_image.data, raw_image.rows, raw_image.cols);

    test_fast_detector(&image, feature_num_need);
    test_harris_detector(&image, feature_num_need);
    test_cv_good_feature(&image, feature_num_need);
    cv::waitKey(0);

    return 0;
}