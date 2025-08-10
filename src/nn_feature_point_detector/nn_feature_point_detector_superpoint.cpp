#include "nn_feature_point_detector.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "tick_tock.h"

namespace FEATURE_DETECTOR {

template <>
bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptor(const GrayImage &image,
                                                              std::vector<Vec2> &all_pixel_uv,
                                                              std::vector<SuperpointDescriptorType> &descriptors) {
    RETURN_FALSE_IF(!session_);

    if (!InferenceSession(image)) {
        ReportError("[NNFeaturePointDetector] Failed to infer session.");
        return false;
    }

    switch (options_.kModelType) {
        case ModelType::kSuperpoint:
            return DetectGoodFeaturesWithDescriptorBySuperpoint(image, all_pixel_uv, descriptors);
        case ModelType::kSuperpointNms:
            return DetectGoodFeaturesWithDescriptorBySuperpointNms(image, all_pixel_uv, descriptors);
        default:
            return false;
    }

    return true;
}

bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptorBySuperpoint(const GrayImage &image,
                                                                          std::vector<Vec2> &all_pixel_uv,
                                                                          std::vector<SuperpointDescriptorType> &descriptors) {
    // Create mask firstly.
    if (!CreateMask(image, all_pixel_uv)) {
        ReportError("[NNFeaturePointDetector] Failed to create mask.");
        return false;
    }

    // Process model output.
    std::vector<Eigen::Map<const MatImgF>> scores_matrices;
    if (!OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[0], scores_matrices)) {
        ReportError("[NNFeaturePointDetector] Failed to convert score tensor value to image matrices.");
        return false;
    }
    std::vector<Eigen::Map<const MatImgF>> descriptors_matrices;
    if (!OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[1], descriptors_matrices)) {
        ReportError("[NNFeaturePointDetector] Failed to convert descriptor tensor value to image matrices.");
        return false;
    }

    // Select candidates from heatmap.
    if (!SelectKeypointCandidatesFromHeatMap(scores_matrices[0])) {
        ReportError("[NNFeaturePointDetector] Failed to select keypoint candidates from heatmap.");
        return false;
    }

    // Select good features from candidates with mask.
    if (!SelectGoodFeaturesFromCandidates(all_pixel_uv)) {
        ReportError("[NNFeaturePointDetector] Failed to select good features from candidates.");
        return false;
    }

    return true;
}

bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptorBySuperpointNms(const GrayImage &image,
                                                                             std::vector<Vec2> &all_pixel_uv,
                                                                             std::vector<SuperpointDescriptorType> &descriptors) {

    return true;
}

}
