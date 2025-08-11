#include "nn_feature_point_detector.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"

namespace FEATURE_DETECTOR {

Ort::Env NNFeaturePointDetector::onnx_environment_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NNFeaturePointDetector");

bool NNFeaturePointDetector::Initialize() {
    const std::string model_root_path = "../onnx_models/";
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
    OnnxRuntime::ConvertImageToTensor(image, memory_info_, input_tensor_);
    run_options_.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL_WARNING);

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
    features.clear();
    features.reserve(options_.kMaxNumberOfDetectedFeatures);
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

} // End of namespace FEATURE_DETECTOR.
