#include "nn_feature_point_detector.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "tick_tock.h"

namespace FEATURE_DETECTOR {

template <>
bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptor(const GrayImage &image, std::vector<Vec2> &all_pixel_uv,
                                                              std::vector<DiskDescriptorType> &descriptors) {
    RETURN_FALSE_IF(!session_);

    if (!InferenceSession(image)) {
        ReportError("[NNFeaturePointDetector] Failed to infer session.");
        return false;
    }

    switch (options_.kModelType) {
        case ModelType::kDiskHeatmap:
            return DetectGoodFeaturesWithDescriptorByDisk(image, all_pixel_uv, descriptors);
        case ModelType::kDiskNms:
            return DetectGoodFeaturesWithDescriptorByDiskNms(image, all_pixel_uv, descriptors);
        default:
            return false;
    }

    return true;
}

bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptorByDisk(const GrayImage &image, std::vector<Vec2> &all_pixel_uv,
                                                                    std::vector<DiskDescriptorType> &descriptors) {
    if (output_tensors_.size() != 2) {
        ReportError("[NNFeaturePointDetector] Model kDiskHeatmap error: output tensors size is not 2.");
        return false;
    }

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

    // Extract descriptors for selected features.
    if (!ExtractDescriptorsForSelectedFeatures(all_pixel_uv, descriptors_matrices, descriptors)) {
        ReportError("[NNFeaturePointDetector] Failed to extract descriptors for selected features.");
        return false;
    }

    return true;
}

bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptorByDiskNms(const GrayImage &image, std::vector<Vec2> &all_pixel_uv,
                                                                       std::vector<DiskDescriptorType> &descriptors) {
    if (output_tensors_.size() != 3) {
        ReportError("[NNFeaturePointDetector] Model kDiskNms error: output tensors size is not 3.");
        return false;
    }

    // Create mask firstly.
    if (!CreateMask(image, all_pixel_uv)) {
        ReportError("[NNFeaturePointDetector] Failed to create mask.");
        return false;
    }

    // Process model output.
    std::vector<Eigen::Map<const TMatImg<int64_t>>> candidates_keypoints_matrices;
    if (!OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[0], candidates_keypoints_matrices)) {
        ReportError("[NNFeaturePointDetector] Failed to convert candidate keypoints tensor value to matrices.");
        return false;
    }
    std::vector<Eigen::Map<const MatImgF>> scores_matrices;
    if (!OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[1], scores_matrices)) {
        ReportError("[NNFeaturePointDetector] Failed to convert score tensor value to matrices.");
        return false;
    }
    std::vector<Eigen::Map<const MatImgF>> descriptors_matrices;
    if (!OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[2], descriptors_matrices)) {
        ReportError("[NNFeaturePointDetector] Failed to convert descriptor tensor value to matrices.");
        return false;
    }

    std::vector<int32_t> sorted_indices;
    SlamOperation::ArgSort(scores_matrices[0].data(), scores_matrices[0].cols(), sorted_indices);
    DirectlySelectGoodFeaturesWithDescriptors(candidates_keypoints_matrices[0], scores_matrices[0], descriptors_matrices[0], sorted_indices, all_pixel_uv,
                                              descriptors);

    return true;
}

}  // namespace FEATURE_DETECTOR
