#ifndef _NN_FEATURE_POINT_DETECTOR_H_
#define _NN_FEATURE_POINT_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "onnx_run_time.h"
#include "map"

namespace FEATURE_DETECTOR {

using SuperpointDescriptorType = Eigen::Matrix<float, 256, 1>;

/* Class NNFeaturePointDetector Declaration. */
class NNFeaturePointDetector {

public:
    enum class ModelType : uint8_t {
        kSuperpoint = 0,
        kSuperpointNms = 1,
        kDisk = 2,
        kDiskNms = 3,
    };

    struct Options {
        int32_t kInvalidBoundary = 3;
        int32_t kMinFeatureDistance = 15;
        int32_t kMaxImageRows = 480;
        int32_t kMaxImageCols = 752;
        int32_t kMaxNumberOfDetectedFeatures = 240;
        float kMinResponse = 0.1f;
        ModelType kModelType = ModelType::kSuperpoint;
        bool kComputeDescriptors = false;
    };

public:
    NNFeaturePointDetector() = default;
    virtual ~NNFeaturePointDetector() = default;

    bool Initialize();
    template <typename NNFeatureDescriptorType>
    bool DetectGoodFeaturesWithDescriptor(const GrayImage &image,
                                          std::vector<Vec2> &all_pixel_uv,
                                          std::vector<NNFeatureDescriptorType> &descriptors);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool InferenceSession(const GrayImage &image);
    bool CreateMask(const GrayImage &image, std::vector<Vec2> &features);
    void DrawRectangleInMask(const int32_t row, const int32_t col, const int32_t radius);
    void UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features);

    bool DetectGoodFeaturesWithDescriptorBySuperpoint(const GrayImage &image,
                                                      std::vector<Vec2> &all_pixel_uv,
                                                      std::vector<SuperpointDescriptorType> &descriptors);
    bool DetectGoodFeaturesWithDescriptorBySuperpointNms(const GrayImage &image,
                                                         std::vector<Vec2> &all_pixel_uv,
                                                         std::vector<SuperpointDescriptorType> &descriptors);

    bool SelectKeypointCandidatesFromHeatMap(const MatImgF &heatmap);
    bool SelectGoodFeaturesFromCandidates(std::vector<Vec2> &features);
    template <typename NNFeatureDescriptorType>
    bool ExtractDescriptorsForSelectedFeatures(const std::vector<Vec2> &features,
                                               const std::vector<Eigen::Map<const MatImgF>> &descriptors_matrices,
                                               std::vector<NNFeatureDescriptorType> &descriptors);

private:
    Options options_;

    static Ort::Env onnx_environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::RunOptions run_options_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    OnnxRuntime::ImageTensor input_tensor_;
    std::vector<Ort::Value> output_tensors_;

    std::multimap<float, Pixel> candidates_;
    MatImg mask_;
};

} // namespace FEATURE_DETECTOR

#endif // end of _NN_FEATURE_POINT_DETECTOR_H_
