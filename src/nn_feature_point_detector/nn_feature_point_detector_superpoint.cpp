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

template <>
bool NNFeaturePointDetector::DirectlySelectGoodFeaturesWithDescriptors(const Ort::Value &candidates_pixel_uv,
                                                        const Eigen::Map<const MatImgF> &candidates_score,
                                                        const Eigen::Map<const MatImgF> &candidates_descriptor,
                                                        const std::vector<int32_t> sorted_indices,
                                                        std::vector<Vec2> &all_pixel_uv,
                                                        std::vector<SuperpointDescriptorType> &descriptors) {
    const auto &tensor_info = candidates_pixel_uv.GetTensorTypeAndShapeInfo();
    const auto &element_type = tensor_info.GetElementType();
    RETURN_FALSE_IF(element_type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    const std::vector<int64_t> &tensor_dims = tensor_info.GetShape();
    const int64_t *data = reinterpret_cast<const int64_t *>(candidates_pixel_uv.GetTensorRawData());
    const int32_t step = tensor_dims[tensor_dims.size() - 1];

    // Extract features with high score.
    all_pixel_uv.reserve(options_.kMaxNumberOfDetectedFeatures);
    std::vector<int32_t> all_pixel_indices;
    all_pixel_indices.reserve(options_.kMaxNumberOfDetectedFeatures);
    for (auto it = sorted_indices.rbegin(); it != sorted_indices.rend(); ++it) {
        const int32_t index = *it;
        const TVec2<int32_t> pixel_uv = TVec2<int32_t>(
            data[index * step], data[index * step + 1]
        );
        CONTINUE_IF(!mask_(pixel_uv.y(), pixel_uv.x()));
        all_pixel_uv.emplace_back(pixel_uv.cast<float>());
        all_pixel_indices.emplace_back(index);
        BREAK_IF(all_pixel_uv.size() >= static_cast<uint32_t>(options_.kMaxNumberOfDetectedFeatures));
        DrawRectangleInMask(pixel_uv.y(), pixel_uv.x(), options_.kMinFeatureDistance);
    }

    // Extract selected descriptors.
    descriptors.resize(all_pixel_indices.size());
    for (uint32_t i = 0; i < descriptors.size(); ++i) {
        descriptors[i] = candidates_descriptor.row(all_pixel_indices[i]).transpose();
    }

    return true;
}

bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptorBySuperpoint(const GrayImage &image,
                                                                          std::vector<Vec2> &all_pixel_uv,
                                                                          std::vector<SuperpointDescriptorType> &descriptors) {
    RETURN_FALSE_IF(output_tensors_.size() != 2);

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
    RETURN_FALSE_IF(output_tensors_.size() != 3);

    // Create mask firstly.
    if (!CreateMask(image, all_pixel_uv)) {
        ReportError("[NNFeaturePointDetector] Failed to create mask.");
        return false;
    }

    // Process model output.
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
    DirectlySelectGoodFeaturesWithDescriptors(output_tensors_[0], scores_matrices[0], descriptors_matrices[0], sorted_indices, all_pixel_uv, descriptors);

    return true;
}

}
