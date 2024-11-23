#ifndef _FEATURE_DESCRIPTOR_BRIEF_H_
#define _FEATURE_DESCRIPTOR_BRIEF_H_

#include "basic_type.h"
#include "datatype_image.h"

#include "descriptor.h"

namespace FEATURE_DETECTOR {

struct BriefOptions {
    int32_t kLength = 256;
    int32_t kValidBoundary = 16;
    int32_t kHalfPatchSize = 8;
};

using BriefType = std::vector<bool>;

/* Class Descriptor Declaration. */
class BriefDescriptor : public Descriptor<BriefOptions, BriefType> {

public:
    BriefDescriptor() : Descriptor<BriefOptions, BriefType>() {}
    virtual ~BriefDescriptor() = default;

private:
    virtual bool ComputeForOneFeature(const GrayImage &image,
                                      const Vec2 &pixel_uv,
                                      BriefType &descriptor) override;

private:
    // All indice of [drow1, dcol1, drow2, dcol2].
    static std::array<int16_t, 256 * 4> pattern_idx_;
};

}

#endif // end of _FEATURE_DESCRIPTOR_BRIEF_H_
