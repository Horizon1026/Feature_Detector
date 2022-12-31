#ifndef _FEATURE_FAST_H_
#define _FEATURE_FAST_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

class FastFeature {

public:
    explicit FastFeature() = default;
    virtual ~FastFeature() = default;

    float ComputeResponse(const Image *image,
                          const int32_t row,
                          const int32_t col);

private:

};

}

#endif
