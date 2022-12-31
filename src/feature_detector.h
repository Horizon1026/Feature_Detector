#ifndef _FEATURE_DETECTOR_H_
#define _FEATURE_DETECTOR_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class FeatureDetector {

public:
    struct Pixel {
        int32_t row = 0;
        int32_t col = 0;
    };

    enum FeatureDetectMethod: uint8_t {
        FAST = 0,
        HARRIS,
        SHI_TOMAS,
    };

    struct FeatureDetectOptions {
        float kMinValidResponse = 20.0f;
        int32_t kMinFeatureDistance = 10;
        FeatureDetectMethod kMethod = HARRIS;
    };

public:
    explicit FeatureDetector() = default;
    virtual ~FeatureDetector() = default;

    void DetectGoodFeatures(const Image *image,
                            const int32_t needed_feature_num,
                            std::vector<Pixel> &features);

private:
    bool SelectCandidates(const Image *image);

    bool SelectGoodFeatures(const Image *image);

private:
    std::map<float, Pixel> candidates_;
    FeatureDetectOptions options_;
    Image mask;     // Used for sparse features.

};

}

#endif
