#ifndef _NN_FEATURE_POINT_DETECTOR_H_
#define _NN_FEATURE_POINT_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "libtorch.h"

namespace FEATURE_DETECTOR {

using XFeatDescriptorType = std::array<float, 64>;

/* Class NNFeaturePointDetector Declaration. */
class NNFeaturePointDetector {

public:
    enum class ModelType : uint8_t {
        kXFeat = 0,
        kSuperpoint = 1,
        kAlike = 2,
    };

    struct Options {
        int32_t kInvalidBoundary = 8;
        int32_t kMinFeatureDistance = 5;
        int32_t kGridFilterRowDivideNumber = 15;
        int32_t kGridFilterColDivideNumber = 15;
        float kMinResponse = 0.5f;
        ModelType kModelType = ModelType::kXFeat;
        bool kComputeDescriptors = true;
    };

public:
    NNFeaturePointDetector() = delete;
    explicit NNFeaturePointDetector(const std::string &model_path);
    virtual ~NNFeaturePointDetector() = default;

    bool ReloadModel(const std::string &model_path);

    template <typename NNFeatureDescriptorType>
    bool DetectGoodFeaturesWithDescriptor(const GrayImage &image,
                                          const uint32_t needed_feature_num,
                                          std::vector<Vec2> &features,
                                          std::vector<NNFeatureDescriptorType> &descriptors);

public:
    // Reference for member variables.
    Options &options() { return options_; }
    MatImgF &keypoints_heat_map() { return keypoints_heat_map_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }
    const MatImgF &keypoints_heat_map() const { return keypoints_heat_map_; }

private:
    // General operations.
    template <typename NNFeatureDescriptorType>
    bool CheckModelInput(const GrayImage &image,
                         const uint32_t needed_feature_num,
                         std::vector<Vec2> &features,
                         std::vector<NNFeatureDescriptorType> &descriptors);
    template <typename NNFeatureDescriptorType>
    bool Preparation(const GrayImage &image,
                     const uint32_t needed_feature_num,
                     std::vector<Vec2> &features,
                     std::vector<NNFeatureDescriptorType> &descriptors);
    void DrawRectangleInMask(const int32_t row, const int32_t col, const int32_t radius);
    void UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features);
    bool SelectKeypointCandidatesFromHeatMap();
    bool SelectGoodFeaturesFromCandidates(const uint32_t needed_feature_num, std::vector<Vec2> &features);
    template <typename NNFeatureDescriptorType>
    bool ExtractDescriptorsForSelectedFeatures(const std::vector<Vec2> &features, std::vector<NNFeatureDescriptorType> &descriptors);

    // Model operations.
    bool ExecuteModel(const GrayImage &image);
    bool ProcessModelOutputXFeat();
    bool ProcessModelOutputSuperpoint();
    bool ProcessModelOutputAlike();

private:
    // Options.
    Options options_;

    // Model related.
    torch::jit::script::Module nn_model_;
    struct ModelInput {
        std::vector<torch::jit::IValue> jit;
        torch::Tensor tensor;
    } model_input_;
    struct ModelOutput {
        torch::jit::IValue raw;
        torch::Tensor descriptors;
        torch::Tensor keypoints;
        torch::Tensor reliability;
    } model_output_;

    // Post process related.
    MatImgF keypoints_heat_map_;
    MatImg mask_;
    std::multimap<uint8_t, Pixel> candidates_;
};

/* Class NNFeaturePointDetector Implementation. */
template <typename NNFeatureDescriptorType>
bool NNFeaturePointDetector::CheckModelInput(const GrayImage &image,
                                             const uint32_t needed_feature_num,
                                             std::vector<Vec2> &features,
                                             std::vector<NNFeatureDescriptorType> &descriptors) {
    RETURN_FALSE_IF(image.rows() == 0 || image.cols() == 0);
    RETURN_FALSE_IF(image.data() == nullptr);
    RETURN_FALSE_IF(needed_feature_num == 0);
    if (!features.empty()) {
        RETURN_FALSE_IF(features.size() != descriptors.size());
    }
    return true;
}

template <typename NNFeatureDescriptorType>
bool NNFeaturePointDetector::Preparation(const GrayImage &image,
                                         const uint32_t needed_feature_num,
                                         std::vector<Vec2> &features,
                                         std::vector<NNFeatureDescriptorType> &descriptors) {
    keypoints_heat_map_.resize(image.rows(), image.cols());
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

} // namespace FEATURE_DETECTOR

#endif // end of _NN_FEATURE_POINT_DETECTOR_H_
