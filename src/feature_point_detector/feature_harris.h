#ifndef _FEATURE_HARRIS_H_
#define _FEATURE_HARRIS_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "feature_point.h"

namespace FEATURE_DETECTOR {

struct HarrisOptions {
    float k = 0.0f;
    int32_t kHalfPatchSize = 1;
    float kMinValidResponse = 0.1f;
};

class HarrisFeature : public Feature<HarrisOptions> {

public:
    HarrisFeature() : Feature<HarrisOptions>() {}
    virtual ~HarrisFeature() = default;

    bool ComputeGradient(const Image &image);

    virtual bool SelectAllCandidates(const Image &image,
                                     const MatInt &mask,
                                     std::map<float, Pixel> &candidates) override;

    virtual float ComputeResponse(const Image &image,
                                  const int32_t row,
                                  const int32_t col) override;

private:
    Mat Ix_;
    Mat Iy_;

};

}

#endif // end of _FEATURE_HARRIS_H_
