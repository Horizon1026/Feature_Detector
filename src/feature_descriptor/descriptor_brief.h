#ifndef _FEATURE_DESCRIPTOR_BRIEF_H_
#define _FEATURE_DESCRIPTOR_BRIEF_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "descriptor.h"

namespace feature_detector {

using BriefType = std::vector<bool>;

/* Class Descriptor Declaration. */
class BriefDescriptor: public Descriptor<BriefType> {

public:
    struct Options {
        int32_t kLength = 256;
        int32_t kHalfPatchSize = 8;
    };

public:
    BriefDescriptor(): Descriptor<BriefType>() {}
    virtual ~BriefDescriptor() = default;

    // Reference for member variables.
    Options &options() { return options_; }
    const Options &options() const { return options_; }

private:
    virtual bool ComputeForOneFeature(const GrayImage &image, const Vec2 &pixel_uv, BriefType &descriptor) const override;

private:
    Options options_;
    // All indice of [drow1, dcol1, drow2, dcol2].
    static std::array<int16_t, 256 * 4> pattern_idx_;
};

}  // namespace feature_detector

#endif  // end of _FEATURE_DESCRIPTOR_BRIEF_H_
