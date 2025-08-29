#include "nn_feature_point_detector.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"

namespace FEATURE_DETECTOR {

Ort::Env NNFeaturePointDetector::onnx_environment_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NNFeaturePointDetector");

bool NNFeaturePointDetector::Initialize() {
    const std::string model_root_path = "../../Feature_Detector/onnx_models/";
    std::string model_path;
    switch (options_.kModelType) {
        default:
        case ModelType::kSuperpoint: {
            model_path = model_root_path + "superpoint.onnx";
            break;
        }
        case ModelType::kSuperpointNms: {
            model_path = model_root_path + "superpoint_nms.onnx";
            break;
        }
        case ModelType::kDisk: {
            model_path = model_root_path + "disk.onnx";
            break;
        }
        case ModelType::kDiskNms: {
            model_path = model_root_path + "disk_nms.onnx";
            break;
        }
    }

    // Initialize session options if needed.
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // For onnxruntime of version 1.20, enable GPU.
    OnnxRuntime::TryToEnableCuda(session_options_);

    // Create session.
    try {
        session_ = Ort::Session(NNFeaturePointDetector::onnx_environment_, model_path.c_str(), session_options_);
        OnnxRuntime::ReportInformationOfSession(session_);
        ReportInfo("[NNFeaturePointDetector] Succeed to load onnx model: " << model_path);
    } catch (const Ort::Exception &e) {
        ReportError("[NNFeaturePointDetector] Failed to load onnx model: " << model_path);
    }
    OnnxRuntime::GetSessionIO(session_, input_names_, output_names_);
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // Infer session once.
    MatImg random_image_matrix = MatImg::Ones(options_.kMaxImageRows, options_.kMaxImageCols);
    const GrayImage random_image(random_image_matrix.data(), random_image_matrix.rows(), random_image_matrix.cols(), false);
    InferenceSession(random_image);

    return true;
}

bool NNFeaturePointDetector::CreateMask(const GrayImage &image, std::vector<Vec2> &features) {
    mask_.resize(image.rows(), image.cols());
    mask_.setConstant(1);
    if (options_.kInvalidBoundary) {
        mask_.topRows(options_.kInvalidBoundary).setZero();
        mask_.bottomRows(options_.kInvalidBoundary).setZero();
        mask_.leftCols(options_.kInvalidBoundary).setZero();
        mask_.rightCols(options_.kInvalidBoundary).setZero();
    }
    if (!features.empty()) {
        UpdateMaskByFeatures(image, features);
    }
    return true;
}

void NNFeaturePointDetector::DrawRectangleInMask(const int32_t row, const int32_t col, const int32_t radius) {
    const int32_t row_start = std::max(0, row - radius);
    const int32_t row_end = std::min(static_cast<int32_t>(mask_.rows() - 1), row + radius);
    const int32_t col_start = std::max(0, col - radius);
    const int32_t col_end = std::min(static_cast<int32_t>(mask_.cols() - 1), col + radius);

    mask_.block(row_start, col_start, row_end - row_start + 1, col_end - col_start + 1).setZero();
}

void NNFeaturePointDetector::UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features) {
    for (const auto &feature: features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col, options_.kMinFeatureDistance);
    }
}

bool NNFeaturePointDetector::InferenceSession(const GrayImage &image) {
    RETURN_FALSE_IF(!session_);

    // Prepare input tensor.
    switch (options_.kModelType) {
        case ModelType::kDisk:
        case ModelType::kDiskNms:
            OnnxRuntime::ConvertGrayImageToRgbTensor(image, memory_info_, input_tensor_);
            break;
        default:
        case ModelType::kSuperpoint:
        case ModelType::kSuperpointNms:
            OnnxRuntime::ConvertImageToTensor(image, memory_info_, input_tensor_);
            break;
    }

    // Prepare run options.
    run_options_.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL_WARNING);

    // Prepare input and output names.
    std::vector<const char *> input_names_ptr_;
    std::vector<const char *> output_names_ptr_;
    input_names_ptr_.reserve(input_names_.size());
    output_names_ptr_.reserve(output_names_.size());
    for (const auto &name: input_names_) {
        input_names_ptr_.emplace_back(name.c_str());
    }
    for (const auto &name: output_names_) {
        output_names_ptr_.emplace_back(name.c_str());
    }

    // Infer session.
    output_tensors_ = session_.Run(run_options_,
        input_names_ptr_.data(), &input_tensor_.value, input_names_ptr_.size(),
        output_names_ptr_.data(), output_names_ptr_.size());
    return true;
}

bool NNFeaturePointDetector::SelectKeypointCandidatesFromHeatMap(const MatImgF &heatmap) {
    candidates_.clear();
    for (int32_t row = 0; row < heatmap.rows(); ++row) {
        for (int32_t col = 0; col < heatmap.cols(); ++col) {
            const float response = heatmap(row, col);
            if (response > options_.kMinResponse) {
                candidates_.insert(std::make_pair(response, Pixel(col, row)));
            }
        }
    }
    return true;
}

bool NNFeaturePointDetector::SelectGoodFeaturesFromCandidates(std::vector<Vec2> &features) {
    if (features.empty()) {
        features.reserve(options_.kMaxNumberOfDetectedFeatures);
    }
    for (auto it = candidates_.crbegin(); it != candidates_.crend(); ++it) {
        const Pixel pixel = it->second;
        const int32_t row = pixel.y();
        const int32_t col = pixel.x();
        CONTINUE_IF(!mask_(row, col));
        features.emplace_back(Vec2(pixel.x(), pixel.y()));
        BREAK_IF(features.size() >= static_cast<uint32_t>(options_.kMaxNumberOfDetectedFeatures));
        DrawRectangleInMask(row, col, options_.kMinFeatureDistance);
    }
    return true;
}


template bool NNFeaturePointDetector::ExtractDescriptorsForSelectedFeatures<SuperpointDescriptorType>(const std::vector<Vec2> &features,
    const std::vector<Eigen::Map<const MatImgF>> &descriptors_matrices, std::vector<SuperpointDescriptorType> &descriptors);
template bool NNFeaturePointDetector::ExtractDescriptorsForSelectedFeatures<DiskDescriptorType>(const std::vector<Vec2> &features,
    const std::vector<Eigen::Map<const MatImgF>> &descriptors_matrices, std::vector<DiskDescriptorType> &descriptors);
template <typename NNFeatureDescriptorType>
bool NNFeaturePointDetector::ExtractDescriptorsForSelectedFeatures(const std::vector<Vec2> &features,
                                                                   const std::vector<Eigen::Map<const MatImgF>> &descriptors_matrices,
                                                                   std::vector<NNFeatureDescriptorType> &descriptors) {
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
        NNFeatureDescriptorType &descriptor = descriptors[i];
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

template bool NNFeaturePointDetector::DirectlySelectGoodFeaturesWithDescriptors<SuperpointDescriptorType>(const Eigen::Map<const TMatImg<int64_t>> &candidates_pixel_uv,
    const Eigen::Map<const MatImgF> &candidates_score, const Eigen::Map<const MatImgF> &candidates_descriptor, const std::vector<int32_t> sorted_indices,
    std::vector<Vec2> &all_pixel_uv, std::vector<SuperpointDescriptorType> &descriptors);
template bool NNFeaturePointDetector::DirectlySelectGoodFeaturesWithDescriptors<DiskDescriptorType>(const Eigen::Map<const TMatImg<int64_t>> &candidates_pixel_uv,
    const Eigen::Map<const MatImgF> &candidates_score, const Eigen::Map<const MatImgF> &candidates_descriptor, const std::vector<int32_t> sorted_indices,
    std::vector<Vec2> &all_pixel_uv, std::vector<DiskDescriptorType> &descriptors);
template <typename NNFeatureDescriptorType>
bool NNFeaturePointDetector::DirectlySelectGoodFeaturesWithDescriptors(const Eigen::Map<const TMatImg<int64_t>> &candidates_pixel_uv,
                                                                       const Eigen::Map<const MatImgF> &candidates_score,
                                                                       const Eigen::Map<const MatImgF> &candidates_descriptor,
                                                                       const std::vector<int32_t> sorted_indices,
                                                                       std::vector<Vec2> &all_pixel_uv,
                                                                       std::vector<NNFeatureDescriptorType> &descriptors) {
    // Extract features with high score.
    all_pixel_uv.reserve(options_.kMaxNumberOfDetectedFeatures);
    std::vector<int32_t> all_pixel_indices;
    all_pixel_indices.reserve(options_.kMaxNumberOfDetectedFeatures);
    for (auto it = sorted_indices.rbegin(); it != sorted_indices.rend(); ++it) {
        const int32_t index = *it;
        const TVec2<int32_t> pixel_uv = candidates_pixel_uv.row(index).cast<int32_t>().transpose();
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

} // End of namespace FEATURE_DETECTOR.
