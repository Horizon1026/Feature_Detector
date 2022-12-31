#ifndef _FEATURE_HARRIS_H_
#define _FEATURE_HARRIS_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class HarrisFeature {

public:
    explicit HarrisFeature() = default;
    virtual ~HarrisFeature() = default;

    float ComputeResponse(const Image *image,
                          const int32_t row,
                          const int32_t col);

private:

};

}

#endif
