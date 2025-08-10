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
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

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

    for (int32_t r = row_start; r <= row_end; ++r) {
        for (int32_t c = col_start; c <= col_end; ++c) {
            mask_(r, c) = 0;
        }
    }
}

void NNFeaturePointDetector::UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features) {
    for (const auto &feature: features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col, options_.kMinFeatureDistance);
    }
}

bool NNFeaturePointDetector::InferenceSession(const GrayImage &image) {
    OnnxRuntime::ConvertImageToTensor(image, memory_info_, input_tensor_);
    run_options_.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL_WARNING);

    return true;
}

} // End of namespace FEATURE_DETECTOR.
