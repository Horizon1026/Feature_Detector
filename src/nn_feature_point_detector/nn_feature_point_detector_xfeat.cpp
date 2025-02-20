#include "nn_feature_point_detector.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"

namespace FEATURE_DETECTOR {

bool NNFeaturePointDetector::ExecuteModel(const GrayImage &image) {
    // Convert image to tensor.
    try {
        model_input_.tensor = torch::from_blob(image.data(), {1, 1, image.rows(), image.cols()},
            torch::kByte).to(torch::kFloat).div(255.0);
    } catch (const c10::Error &e) {
        ReportError("[NN Feature Detector] Failed to convert image to tensor.");
        return false;
    }

    // Execute the model.
    try {
        model_input_.jit.clear();
        model_input_.jit.push_back(model_input_.tensor);
        model_output_.raw = nn_model_.forward(model_input_.jit);
    } catch (const c10::Error &e) {
        ReportError("[NN Feature Detector] Failed to execute model.");
        return false;
    }

    return true;
}

bool NNFeaturePointDetector::ProcessModelOutputXFeat() {
    const auto tuple = model_output_.raw.toTuple();
    RETURN_FALSE_IF(tuple->elements().size() != 3);

    // Get keypoints, descriptors and reliability.
    model_output_.descriptors = tuple->elements()[0].toTensor();
    model_output_.keypoints = tuple->elements()[1].toTensor();
    model_output_.reliability = tuple->elements()[2].toTensor();

    // Normalize descriptor tensor.
    // Parameters of norm: (p = 2, dim = 1, keepdim = true).
    model_output_.descriptors = model_output_.descriptors.div(model_output_.descriptors.norm(2, 1, true));

    // Convert keypoints tensor to keypoints heatmap.
    // Do softmax on keypoints tensor and discard the dustbin.
    const torch::Tensor scores = torch::softmax(model_output_.keypoints, /*dim=*/1)
        .slice(/*dim=*/1, /*start=*/0, /*end=*/64);
    const int32_t height = scores.size(2);
    const int32_t width = scores.size(3);
    model_output_.keypoints = scores.permute({0, 2, 3, 1}).reshape({1, height, width, 8, 8});
    model_output_.keypoints = model_output_.keypoints.permute({0, 1, 3, 2, 4}).reshape({1, 1, height * 8, width * 8});
    keypoints_heat_map_.setZero(model_output_.keypoints.size(2), model_output_.keypoints.size(3));
    LibTorch::ConvertToMatImgF(model_output_.keypoints, keypoints_heat_map_);

    // TODO: Process reliability tensor.

    return true;
}

template <>
bool NNFeaturePointDetector::ExtractDescriptorsForSelectedFeatures<XFeatDescriptorType>(const std::vector<Vec2> &features,
                                                                                        std::vector<XFeatDescriptorType> &descriptors) {
    RETURN_TRUE_IF(features.empty());
    RETURN_FALSE_IF(descriptors.size() > features.size());
    const int32_t descriptor_size = model_output_.descriptors.size(1);
    const int32_t tensor_rows = model_output_.descriptors.size(2);
    const int32_t tensor_cols = model_output_.descriptors.size(3);
    RETURN_FALSE_IF(descriptor_size == 0);
    if (descriptors.capacity() < features.size()) {
        descriptors.reserve(features.size());
    }

    // Prelocate address of each layer.
    std::vector<float *> descriptors_ptr(descriptor_size, nullptr);
    for (int32_t j = 0; j < descriptor_size; ++j) {
        descriptors_ptr[j] = model_output_.descriptors[0][j].data_ptr<float>();
    }

    // If descriptors is not empty, do not change the existing part.
    XFeatDescriptorType temp_descriptor;
    for (uint32_t i = descriptors.size(); i < features.size(); ++i) {
        const Vec2 loc_in_tensor = features[i] / 8.0f;

        // Continue if the location is out of the tensor.
        const float &x = loc_in_tensor.x();
        const float &y = loc_in_tensor.y();
        const int32_t x0 = static_cast<int32_t>(x);
        const int32_t y0 = static_cast<int32_t>(y);
        CONTINUE_IF(x0 < 0 || x0 > tensor_cols - 2 || y0 < 0 || y0 > tensor_rows - 2);
        const int32_t x1 = x0 + 1;
        const int32_t y1 = y0 + 1;
        const float dx = x - x0;
        const float dy = y - y0;

        // Bilinear interpolation.
        for (int32_t j = 0; j < descriptor_size; ++j) {
            const float v00 = descriptors_ptr[j][y0 * tensor_cols + x0];
            const float v01 = descriptors_ptr[j][y1 * tensor_cols + x0];
            const float v10 = descriptors_ptr[j][y0 * tensor_cols + x1];
            const float v11 = descriptors_ptr[j][y1 * tensor_cols + x1];
            temp_descriptor[j] = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11;
        }
        descriptors.emplace_back(temp_descriptor);
    }

    RETURN_FALSE_IF(descriptors.size() != features.size());
    return true;
}

template <>
bool NNFeaturePointDetector::DetectGoodFeaturesWithDescriptor<XFeatDescriptorType>(const GrayImage &image,
                                                                                   const uint32_t needed_feature_num,
                                                                                   std::vector<Vec2> &features,
                                                                                   std::vector<XFeatDescriptorType> &descriptors) {
    if (options_.kModelType != ModelType::kXFeat) {
        ReportError("[NN Feature Detector] Model is not xfeat.");
        return false;
    }
    if (!CheckModelInput(image, needed_feature_num, features, descriptors)) {
        ReportError("[NN Feature Detector] Failed to check model input.");
        return false;
    }
    if (!Preparation(image, needed_feature_num, features, descriptors)) {
        ReportError("[NN Feature Detector] Failed to prepare.");
        return false;
    }
    if (!ExecuteModel(image)) {
        ReportError("[NN Feature Detector] Failed to execute model.");
        return false;
    }
    if (!ProcessModelOutputXFeat()) {
        ReportError("[NN Feature Detector] Failed to process model output.");
        return false;
    }
    if (!SelectKeypointCandidatesFromHeatMap()) {
        ReportError("[NN Feature Detector] Failed to select keypoint candidates from heat map.");
        return false;
    }
    if (!SelectGoodFeaturesFromCandidates(needed_feature_num, features)) {
        ReportError("[NN Feature Detector] Failed to select good features from candidates.");
        return false;
    }
    RETURN_TRUE_IF(!options_.kComputeDescriptors);
    if (!ExtractDescriptorsForSelectedFeatures(features, descriptors)) {
        ReportError("[NN Feature Detector] Failed to extract descriptors for selected features.");
        return false;
    }
    return true;
}

} // End of namespace FEATURE_DETECTOR.
