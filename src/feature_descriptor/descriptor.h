#ifndef _FEATURE_DESCRIPTOR_H_
#define _FEATURE_DESCRIPTOR_H_

#include "basic_type.h"
#include "datatype_image.h"

namespace feature_detector {

/* Class Descriptor Declaration. */
template <typename OptionsType, typename DescriptorType>
class Descriptor {

public:
    Descriptor() = default;
    virtual ~Descriptor() = default;

    bool Compute(const GrayImage &image, const std::vector<Vec2> &pixel_uv, std::vector<DescriptorType> &descriptor);

    // Reference for member variables.
    OptionsType &options() { return options_; }

    // Reference for member variables.
    const OptionsType &options() const { return options_; }

private:
    virtual bool ComputeForOneFeature(const GrayImage &image, const Vec2 &pixel_uv, DescriptorType &descriptor) = 0;

private:
    OptionsType options_;
};

/* Class Descriptor Definition. */
template <typename OptionsType, typename DescriptorType>
bool Descriptor<OptionsType, DescriptorType>::Compute(const GrayImage &image, const std::vector<Vec2> &pixel_uv, std::vector<DescriptorType> &descriptor) {
    if (pixel_uv.empty() || image.data() == nullptr) {
        return false;
    }

    if (descriptor.size() != pixel_uv.size()) {
        descriptor.resize(pixel_uv.size());
    }

    const uint32_t max_i = descriptor.size();
    for (uint32_t i = 0; i < max_i; ++i) {
        ComputeForOneFeature(image, pixel_uv[i], descriptor[i]);
    }

    return true;
}

}  // namespace feature_detector

#endif  // end of _FEATURE_DESCRIPTOR_H_
