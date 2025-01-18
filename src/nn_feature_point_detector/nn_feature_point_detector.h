#ifndef _NN_FEATURE_POINT_DETECTOR_H_
#define _NN_FEATURE_POINT_DETECTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "libtorch.h"

namespace FEATURE_DETECTOR {

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

}

#endif // end of _NN_FEATURE_POINT_DETECTOR_H_
