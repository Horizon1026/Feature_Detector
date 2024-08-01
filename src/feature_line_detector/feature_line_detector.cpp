#include "feature_line_detector.h"
#include "slam_operations.h"
#include "log_report.h"

namespace FEATURE_DETECTOR {

bool FeatureLineDetector::DetectGoodFeatures(const GrayImage &image,
                                             const uint32_t needed_feature_num,
                                             std::vector<Vec4> &features) {
    RETURN_FALSE_IF(image.data() == nullptr || image.rows() < 2 || image.cols() < 2);
    features.clear();
    RETURN_TRUE_IF(needed_feature_num == 0);

    RETURN_FALSE_IF_FALSE(ComputeLineLevelAngleMap(image));

    return true;
}

bool FeatureLineDetector::ComputeLineLevelAngleMap(const GrayImage &image)
{
    pixels_.resize(image.rows(), image.cols());

    return true;
}

}
