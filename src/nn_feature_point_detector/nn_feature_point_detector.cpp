#include "nn_feature_point_detector.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"

namespace FEATURE_DETECTOR {

NNFeaturePointDetector::NNFeaturePointDetector(const std::string &model_path) {
    try {
        nn_model_ = torch::jit::load(model_path);
        nn_model_.eval();
    } catch (const c10::Error &e) {
        ReportError("[NN Feature Detector] Failed to load model from: " << model_path);
    }
}

bool NNFeaturePointDetector::DetectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    if (needed_feature_num == 0) {
        return true;
    }

    // Initialize mask.
    mask_.setConstant(image.rows(), image.cols(), 1);
    if (!features.empty()) {
        UpdateMaskByFeatures(image, features);
    }

    // Execute the model.
    RETURN_FALSE_IF(!ExecuteModel(image));

    // Process output of model.
    RETURN_FALSE_IF(!ProcessModelOutputOfDescriptor());
    RETURN_FALSE_IF(!ProcessModelOutputOfKeypoints());
    RETURN_FALSE_IF(!ProcessModelOutputOfReliability());

    // Select good features.
    RETURN_FALSE_IF(!SelectGoodFeatures(needed_feature_num, features));

    return true;
}

void NNFeaturePointDetector::SparsifyFeatures(const std::vector<Vec2> &features,
                                              const int32_t image_rows,
                                              const int32_t image_cols,
                                              const uint8_t status_need_filter,
                                              const uint8_t status_after_filter,
                                              std::vector<uint8_t> &status) {
    if (features.size() != status.size()) {
        status.resize(features.size(), 1);
    }

    // Grid filter to make points sparsely.
    const float grid_row_step = image_rows / (options_.kGridFilterRowDivideNumber - 1);
    const float grid_col_step = image_cols / (options_.kGridFilterColDivideNumber - 1);
    mask_.setConstant(options_.kGridFilterRowDivideNumber, options_.kGridFilterColDivideNumber, 1);
    for (uint32_t i = 0; i < features.size(); ++i) {
        const int32_t row = static_cast<int32_t>(features[i].y() / grid_row_step);
        const int32_t col = static_cast<int32_t>(features[i].x() / grid_col_step);

        if (row < 0 || row > mask_.rows() - 1 || col < 0 || col > mask_.cols() - 1) {
            status[i] = status_after_filter;
            continue;
        }

        if (mask_(row, col) && status[i] == status_need_filter) {
            mask_(row, col) = 0;
        } else if (!mask_(row, col) && status[i] == status_need_filter) {
            status[i] = status_after_filter;
        }
    }
}

bool NNFeaturePointDetector::ExecuteModel(const GrayImage &image) {
    // Convert image to tensor.
    input_ = torch::from_blob(image.data(), {1, 1, image.rows(), image.cols()},
        torch::kByte).to(torch::kFloat).div(255.0);

    // Execute the model and turn its output into a tuple.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_);
    torch::jit::IValue output = nn_model_.forward(inputs);
    auto output_tuple = output.toTuple();

    // Extract each tensor from the tuple.
    model_output_.descriptor = output_tuple->elements()[0].toTensor();
    model_output_.keypoints = output_tuple->elements()[1].toTensor();
    model_output_.reliability = output_tuple->elements()[2].toTensor();

    return true;

}

bool NNFeaturePointDetector::ProcessModelOutputOfDescriptor() {
    // Normalize descriptor tensor.
    // Parameters of norm: (p = 2, dim = 1, keepdim = true).
    model_output_.descriptor = model_output_.descriptor.div(model_output_.descriptor.norm(2, 1, true));
    return true;
}

bool NNFeaturePointDetector::ProcessModelOutputOfKeypoints() {
    // Convert keypoints tensor to keypoints heatmap.
    // Do softmax on keypoints tensor and discard the dustbin.
    const torch::Tensor scores = torch::softmax(model_output_.keypoints, /*dim=*/1)
        .slice(/*dim=*/1, /*start=*/0, /*end=*/64);
    const int32_t height = scores.size(2);
    const int32_t width = scores.size(3);
    model_output_.keypoints = scores.permute({0, 2, 3, 1}).reshape({1, height, width, 8, 8});
    model_output_.keypoints = model_output_.keypoints.permute({0, 1, 3, 2, 4}).reshape({1, 1, height * 8, width * 8});

    keypoints_heat_map_.setZero(model_output_.keypoints.size(2), model_output_.keypoints.size(3));
    LibTorch::ConvertToMatImg(model_output_.keypoints, keypoints_heat_map_);
    return true;
}

bool NNFeaturePointDetector::ProcessModelOutputOfReliability() {
    // TODO:
    return true;
}

bool NNFeaturePointDetector::SelectCandidates() {
    RETURN_FALSE_IF(keypoints_heat_map_.size() == 0);
    candidates_.clear();

    for (int32_t row = 0; row < keypoints_heat_map_.rows(); ++row) {
        for (int32_t col = 0; col < keypoints_heat_map_.cols(); ++col) {
            const float response = keypoints_heat_map_(row, col);
            if (response > options_.kMinResponse) {
                candidates_.emplace(response, Pixel(col, row));
            }
        }
    }

    return true;
}

bool NNFeaturePointDetector::SelectGoodFeatures(const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    RETURN_FALSE_IF(!SelectCandidates());

    for (auto it = candidates_.crbegin(); it != candidates_.crend(); ++it) {
        const Pixel pixel = it->second;
        const int32_t row = pixel.y();
        const int32_t col = pixel.x();
        if (mask_(row, col)) {
            features.emplace_back(Vec2(pixel.x(), pixel.y()));
            RETURN_TRUE_IF(features.size() >= needed_feature_num);
            DrawRectangleInMask(row, col);
        }
    }

    return true;
}

void NNFeaturePointDetector::DrawRectangleInMask(const int32_t row, const int32_t col) {
    const int32_t row_start = std::max(0, row - options_.kMinFeatureDistance);
    const int32_t row_end = std::min(static_cast<int32_t>(mask_.rows() - 1), row + options_.kMinFeatureDistance);
    const int32_t col_start = std::max(0, col - options_.kMinFeatureDistance);
    const int32_t col_end = std::min(static_cast<int32_t>(mask_.cols() - 1), col + options_.kMinFeatureDistance);

    for (int32_t r = row_start; r <= row_end; ++r) {
        for (int32_t c = col_start; c <= col_end; ++c) {
            mask_(r, c) = 0;
        }
    }
}

void NNFeaturePointDetector::UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features) {
    for (const auto &feature : features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col);
    }
}

bool NNFeaturePointDetector::ExtractDescriptors(const std::vector<Vec2> &features, Mat &descriptors) {
    RETURN_TRUE_IF(features.empty());
    const int32_t descriptor_size = model_output_.descriptor.size(1);
    const int32_t tensor_rows = model_output_.descriptor.size(2);
    const int32_t tensor_cols = model_output_.descriptor.size(3);
    RETURN_FALSE_IF(descriptor_size == 0);
    descriptors.setZero(features.size(), descriptor_size);

    for (uint32_t i = 0; i < features.size(); ++i) {
        const Vec2 loc_in_tensor = features[i] / 8.0f;
        // Continue if the location is out of the tensor.
        CONTINUE_IF(loc_in_tensor.x() < 0 || loc_in_tensor.x() > tensor_cols - 2 || loc_in_tensor.y() < 0 || loc_in_tensor.y() > tensor_rows - 2);
        for (int32_t j = 0; j < descriptor_size; ++j) {
            // Bilinear interpolation.
            const float x = loc_in_tensor.x();
            const float y = loc_in_tensor.y();
            const int32_t x0 = static_cast<int32_t>(x);
            const int32_t y0 = static_cast<int32_t>(y);
            const int32_t x1 = x0 + 1;
            const int32_t y1 = y0 + 1;
            const float dx = x - x0;
            const float dy = y - y0;
            const float v00 = model_output_.descriptor[0][j][y0][x0].item<float>();
            const float v01 = model_output_.descriptor[0][j][y1][x0].item<float>();
            const float v10 = model_output_.descriptor[0][j][y0][x1].item<float>();
            const float v11 = model_output_.descriptor[0][j][y1][x1].item<float>();
            descriptors(i, j) = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11;
        }
    }

    return true;
}

} // End of namespace FEATURE_DETECTOR.
