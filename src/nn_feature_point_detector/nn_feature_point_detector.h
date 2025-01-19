#ifndef _NN_FEATURE_POINT_DETECTOR_H_
#define _NN_FEATURE_POINT_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "libtorch.h"

namespace FEATURE_DETECTOR {

using XFeatType = std::array<float, 64>;

/* Class NNFeaturePointDetector Declaration. */
class NNFeaturePointDetector {

public:
    struct Options {
        int32_t kMinFeatureDistance = 15;
        int32_t kGridFilterRowDivideNumber = 12;
        int32_t kGridFilterColDivideNumber = 12;
        uint8_t kMinResponse = 25;
    };

public:
    NNFeaturePointDetector() = delete;
    explicit NNFeaturePointDetector(const std::string &model_path);
    virtual ~NNFeaturePointDetector() = default;

    bool DetectGoodFeatures(const GrayImage &image,
                            const uint32_t needed_feature_num,
                            std::vector<Vec2> &features);

    void SparsifyFeatures(const std::vector<Vec2> &features,
                          const int32_t image_rows,
                          const int32_t image_cols,
                          const uint8_t status_need_filter,
                          const uint8_t status_after_filter,
                          std::vector<uint8_t> &status);

    bool ExtractDescriptors(const std::vector<Vec2> &features,
                            Mat &descriptors);

    template <uint32_t DescriptorSize>
    bool ExtractDescriptors(const std::vector<Vec2> &features,
                            std::vector<std::array<float, DescriptorSize>> &descriptors);

public:
    // Reference for member variables.
    Options &options() { return options_; }
    MatImg &keypoints_heat_map() { return keypoints_heat_map_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }
    const MatImg &keypoints_heat_map() const { return keypoints_heat_map_; }

private:
    bool ExecuteModel(const GrayImage &image);
    bool ProcessModelOutputOfDescriptor();
    bool ProcessModelOutputOfKeypoints();
    bool ProcessModelOutputOfReliability();

    bool SelectCandidates();
    bool SelectGoodFeatures(const uint32_t needed_feature_num,
                            std::vector<Vec2> &features);
    void DrawRectangleInMask(const int32_t row,
                             const int32_t col);
    void UpdateMaskByFeatures(const GrayImage &image,
                              const std::vector<Vec2> &features);

private:
    Options options_;

    torch::jit::script::Module nn_model_;
    struct ModelOutput {
        torch::Tensor descriptor;
        torch::Tensor keypoints;
        torch::Tensor reliability;
    } model_output_;
    torch::Tensor input_;
    MatImg keypoints_heat_map_;

    std::multimap<uint8_t, Pixel> candidates_;
    MatInt mask_;
};

/* Class NNFeaturePointDetector Definition. */
template <uint32_t DescriptorSize>
bool NNFeaturePointDetector::ExtractDescriptors(const std::vector<Vec2> &features,
                                                std::vector<std::array<float, DescriptorSize>> &descriptors) {
    RETURN_TRUE_IF(features.empty());
    const int32_t descriptor_size = model_output_.descriptor.size(1);
    const int32_t tensor_rows = model_output_.descriptor.size(2);
    const int32_t tensor_cols = model_output_.descriptor.size(3);
    RETURN_FALSE_IF(descriptor_size == 0);
    descriptors.resize(features.size());

    for (uint32_t i = 0; i < features.size(); ++i) {
        const Vec2 loc_in_tensor = features[i] / 8.0f;
        // Continue if the locaiton is out of the tensor.
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
            descriptors[i][j] = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11;
        }
    }

    return true;
}

} // namespace FEATURE_DETECTOR

#endif // end of _NN_FEATURE_POINT_DETECTOR_H_
