#include <FASTFeatureDetector.h>
#include <string>
#include <iostream>
#include <ctime>

clock_t startTime, endTime;

//定义图片存放路径
std::string image_filepath = "../examples/image.png";

int main() {
    FASTFeatureDetectorClass FASTFeatureDetector(0.2, 10, 12, 15, 8);

    // 加载图像（以灰度方式），并检查是否加载成功，如果不成功则终止程序
    cv::Mat image = cv::imread(image_filepath, 0);
    assert(image.data != nullptr);
    cv::imshow("image", image);

    // 检测 FAST 角点
    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    cv::circle(mask, cv::Point2f(image.cols / 2, image.rows / 2), 200, 0, -1);
    startTime = clock();
    std::vector<cv::Point2f> points = FASTFeatureDetector.DetectGoodSparseFeatures(image);
    endTime = clock();
    std::cout << "My code Features detect time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
    std::cout << points.size() << std::endl;

    // OpenCV 检测 FAST 角点
    std::vector<cv::KeyPoint> keypoints;
    startTime = clock();
    cv::FAST(image, keypoints, 40);
    endTime = clock();
    std::cout << "OpenCV Features detect time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;

    // 在原图上画上特征点
    cv::Mat showImage = cv::Mat(image.rows, image.cols, CV_8UC3);
    cv::cvtColor(image, showImage, CV_GRAY2BGR);
    for (auto &point : points) {
        cv::circle(showImage, point, 3, cv::Scalar(0, 0, 255), 1);
    }
    cv::imshow("image with FAST features", showImage);

    // 暂停
    cv::waitKey();

    return 0;
}