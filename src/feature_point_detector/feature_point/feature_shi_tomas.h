#ifndef _FEATURE_SHI_TOMAS_H_
#define _FEATURE_SHI_TOMAS_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "feature_point.h"

namespace FEATURE_DETECTOR {

struct ShiTomasOptions {
    int32_t kHalfPatchSize = 1;
    float kMinValidResponse = 0.1f;
};

class ShiTomasFeature : public Feature<ShiTomasOptions> {

public:
    ShiTomasFeature() : Feature<ShiTomasOptions>() {}
    virtual ~ShiTomasFeature() = default;

    bool ComputeGradient(const GrayImage &image);

    virtual bool SelectAllCandidates(const GrayImage &image,
                                     const MatInt &mask,
                                     std::map<float, Pixel> &candidates) override;

    virtual float ComputeResponse(const GrayImage &image,
                                  const int32_t row,
                                  const int32_t col) override;

private:
    Mat Ix_;
    Mat Iy_;

};

}

#endif // end of _FEATURE_SHI_TOMAS_H_
