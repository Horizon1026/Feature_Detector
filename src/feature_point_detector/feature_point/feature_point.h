#ifndef _FEATURE_POINT_H_
#define _FEATURE_POINT_H_

#include "basic_type.h"
#include "datatype_image.h"

namespace feature_detector {

template <typename OptionsType>
class Feature {

public:
    Feature() = default;
    virtual ~Feature() = default;

    virtual bool SelectAllCandidates(const GrayImage &image, const MatInt &mask, std::map<float, Pixel> &candidates) = 0;

    virtual float ComputeResponse(const GrayImage &image, const int32_t row, const int32_t col) = 0;

    // Reference for member variables.
    OptionsType &options() { return options_; }

    // Const reference for member variables.
    const OptionsType &options() const { return options_; }

private:
    OptionsType options_;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_POINT_H_
