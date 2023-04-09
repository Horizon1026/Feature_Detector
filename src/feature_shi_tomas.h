#ifndef _FEATURE_SHI_TOMAS_H_
#define _FEATURE_SHI_TOMAS_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class ShiTomasFeature {

public:
    struct HarrisOptions {
        int32_t kHalfPatchSize = 1;
    };

public:
    explicit ShiTomasFeature() = default;
    virtual ~ShiTomasFeature() = default;

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

#endif // end of _FEATURE_SHI_TOMAS_H_
