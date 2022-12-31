#include "log_api.h"
#include "feature_detector.h"

#include "opencv2/opencv.hpp"

std::string image_file_path = "../examples/image.png";

void test_harris_detector(int32_t feature_num_need) {

}

void test_fast_detector(int32_t feature_num_need) {

}

void test_cv_good_feature(int32_t feature_num_need) {

}

int main() {
    LogInfo(YELLOW ">> Test feature detector." RESET_COLOR);
    int32_t feature_num_need = 200;

    cv::Mat raw_image = cv::imread(image_file_path);
    cv::imshow("raw_image", raw_image);
    cv::waitKey(0);

    return 0;
}