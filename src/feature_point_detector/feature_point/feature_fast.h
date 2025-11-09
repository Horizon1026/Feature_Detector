#ifndef _FEATURE_FAST_H_
#define _FEATURE_FAST_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "feature_point.h"

namespace feature_detector {

struct FastOptions {
    int32_t kN = 12;  // Fast-12 default.
    int32_t kHalfPatchSize = 3;
    uint8_t kMinPixelDiffValue = 15;
    float kMinValidResponse = 9.0f;
};

class FastFeature : public Feature<FastOptions> {

public:
    FastFeature()
        : Feature<FastOptions>() {}
    virtual ~FastFeature() = default;

    virtual bool SelectAllCandidates(const GrayImage &image, const MatInt &mask, std::map<float, Pixel> &candidates) override;

    virtual float ComputeResponse(const GrayImage &image, const int32_t row, const int32_t col) override;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_FAST_H_
