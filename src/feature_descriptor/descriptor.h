#ifndef _FEATURE_DESCRIPTOR_H_
#define _FEATURE_DESCRIPTOR_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"
#include "type_traits"

namespace feature_detector {

/* Class Descriptor Declaration. */
template <typename DescriptorType>
class Descriptor {

public:
    Descriptor() = default;
    virtual ~Descriptor() = default;
    bool Compute(const GrayImage &image, const std::vector<Vec2> &pixel_uv, std::vector<DescriptorType> &descriptors) const;
    bool Compute(const GrayImage &image, const std::vector<Vec2> &pixel_uv, std::vector<Vec> &descriptors) const;

private:
    virtual bool ComputeForOneFeature(const GrayImage &image, const Vec2 &pixel_uv, DescriptorType &descriptors) const = 0;

};

/* Class Descriptor Definition. */
template <typename DescriptorType>
bool Descriptor<DescriptorType>::Compute(const GrayImage &image, const std::vector<Vec2> &pixel_uv, std::vector<DescriptorType> &descriptors) const {
    RETURN_FALSE_IF(pixel_uv.empty() || image.data() == nullptr);
    if (descriptors.size() != pixel_uv.size()) {
        descriptors.resize(pixel_uv.size());
    }

    const uint32_t max_i = descriptors.size();
    for (uint32_t i = 0; i < max_i; ++i) {
        ComputeForOneFeature(image, pixel_uv[i], descriptors[i]);
    }

    return true;
}

template <typename DescriptorType>
bool Descriptor<DescriptorType>::Compute(const GrayImage &image, const std::vector<Vec2> &pixel_uv, std::vector<Vec> &descriptors) const {
    std::vector<DescriptorType> temp_descriptors;
    RETURN_FALSE_IF(!Compute(image, pixel_uv, temp_descriptors));
    descriptors.resize(temp_descriptors.size());
    for (uint32_t i = 0; i < temp_descriptors.size(); ++i) {
        Vec &descriptor = descriptors[i];
        auto &temp_descriptor = temp_descriptors[i];
        descriptor.resize(temp_descriptor.size());
        if constexpr (std::is_same_v<DescriptorType, std::vector<bool>>) {
            for (uint32_t j = 0; j < temp_descriptor.size(); ++j) {
                descriptor[j] = temp_descriptor[j] ? 1.0f : 0.0f;
            }
        } else {
            for (uint32_t j = 0; j < temp_descriptor.size(); ++j) {
                descriptor[j] = static_cast<float>(temp_descriptor[j]);
            }
        }
    }
    return true;
}

}  // namespace feature_detector

#endif  // end of _FEATURE_DESCRIPTOR_H_
