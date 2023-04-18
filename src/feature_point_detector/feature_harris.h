#ifndef _FEATURE_HARRIS_H_
#define _FEATURE_HARRIS_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class HarrisFeature {

public:
    struct HarrisOptions {
        float k = 0.0f;
        int32_t kHalfPatchSize = 1;
    };

public:
    explicit HarrisFeature() = default;
    virtual ~HarrisFeature() = default;

    bool ComputeGradient(const Image &image);

    float ComputeResponse(const Image &image,
                          const int32_t row,
                          const int32_t col);

    HarrisOptions &options() { return options_; }

    float response() const { return response_; }

private:
    HarrisOptions options_;
    float response_ = 0.0f;
    Mat Ix_;
    Mat Iy_;

};

}

#endif // end of _FEATURE_HARRIS_H_
