#ifndef _FEATURE_POINT_H_
#define _FEATURE_POINT_H_

#include "datatype_basic.h"
#include "datatype_image.h"

namespace FEATURE_DETECTOR {

template<typename OptionsType>
class Feature {

public:
    Feature() = default;
    virtual ~Feature() = default;

    virtual bool SelectAllCandidates(const Image &image,
                                     const MatInt &mask,
                                     std::map<float, Pixel> &candidates) = 0;

    virtual float ComputeResponse(const Image &image,
                                  const int32_t row,
                                  const int32_t col) = 0;

    OptionsType &options() { return options_; }

private:
    OptionsType options_;

};

}

#endif // end of _FEATURE_POINT_H_
