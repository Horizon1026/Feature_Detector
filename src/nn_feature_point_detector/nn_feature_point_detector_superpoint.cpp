#include "nn_feature_point_detector.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "tick_tock.h"

namespace FEATURE_DETECTOR {

template <>
bool NNFeaturePointDetector::ExtractDescriptorsForSelectedFeatures(const std::vector<Vec2> &features,
                                                                   const std::vector<Eigen::Map<const MatImgF>> &descriptors_matrices,
                                                                   std::vector<SuperpointDescriptorType> &descriptors) {
    descriptors.resize(features.size());
    for (uint32_t i = 0; i < descriptors.size(); ++i) {
        const Vec2 &feature = features[i];
        const float row = feature.y() / 8.0f;
        const float col = feature.x() / 8.0f;
        const int32_t int_row = static_cast<int32_t>(row);
        const int32_t int_col = static_cast<int32_t>(col);
        const float sub_row = row - std::floor(row);
        const float sub_col = col - std::floor(col);
        const float inv_sub_row = 1.0f - sub_row;
        const float inv_sub_col = 1.0f - sub_col;
        const std::array<float, 4> weights = {
            inv_sub_col * inv_sub_row,
            sub_col * inv_sub_row,
            inv_sub_col * sub_row,
            sub_col * sub_row
        };

        // using SuperpointDescriptorType = Eigen::Matrix<float, 256, 1>;
        SuperpointDescriptorType &descriptor = descriptors[i];
        for (uint32_t j = 0; j < descriptor.rows(); ++j) {
            const auto &descriptor_map = descriptors_matrices[j];
            const float *map_ptr = descriptor_map.data() + int_row * descriptor_map.cols() + int_col;
            descriptor(j) = static_cast<float>(
                weights[0] * map_ptr[0] + weights[1] * map_ptr[1] +
                weights[2] * map_ptr[descriptor_map.cols()] + weights[3] * map_ptr[descriptor_map.cols() + 1]);
        }
    }
    return true;
}

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

    // Extract descriptors for selected features.
    if (!ExtractDescriptorsForSelectedFeatures(all_pixel_uv, descriptors_matrices, descriptors)) {
        ReportError("[NNFeaturePointDetector] Failed to extract descriptors for selected features.");
        return false;
    }

    return true;
}

bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptorBySuperpointNms(const GrayImage &image,
                                                                             std::vector<Vec2> &all_pixel_uv,
                                                                             std::vector<SuperpointDescriptorType> &descriptors) {
    // TODO:

    return true;
}

}
