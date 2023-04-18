#ifndef _FEATURE_FAST_H_
#define _FEATURE_FAST_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class FastFeature {

public:
    struct FastOptions {
        int32_t kN = 12;        // Fast-12 default.
        int32_t kHalfPatchSize = 3;
        uint8_t kMinPixelDiffValue = 15;
    };

public:
    explicit FastFeature() = default;
    virtual ~FastFeature() = default;

    float ComputeResponse(const Image &image,
                          const int32_t row,
                          const int32_t col);

    float response() const { return response_; }
    FastOptions &options() { return options_; }

private:
    float response_ = 0.0f;
    FastOptions options_;

};

}

#endif // end of _FEATURE_FAST_H_
