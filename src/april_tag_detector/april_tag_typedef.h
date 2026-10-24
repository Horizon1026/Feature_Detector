#ifndef _APRIL_TAG_TYPEDEF_H_
#define _APRIL_TAG_TYPEDEF_H_

#include "basic_type.h"

#include "array"

namespace feature_detector {

enum class AprilTagFamily : uint8_t {
    kUnknown = 0,
    kTagStandard41h12 = 1,
    kTag36H11 = 2,
    kArucoMIP36H12 = 3,
    kAruco4X4_100 = 4,
    kTag16H5 = 5,
    kTag25H9 = 6,
    kTag36H10 = 7,
    kTagCircle21h7 = 8,
    kTagCircle49h12 = 9,
    kTagCustom48h12 = 10,
    kTagStandard52h13 = 11,
    kAruco4X4_50 = 12,
    kAruco4X4_250 = 13,
    kAruco4X4_1000 = 14,
    kAruco5X5_50 = 15,
    kAruco5X5_100 = 16,
    kAruco5X5_250 = 17,
    kAruco5X5_1000 = 18,
    kAruco6X6_50 = 19,
    kAruco6X6_100 = 20,
    kAruco6X6_250 = 21,
    kAruco6X6_1000 = 22,
    kAruco7X7_50 = 23,
    kAruco7X7_100 = 24,
    kAruco7X7_250 = 25,
    kAruco7X7_1000 = 26,
};

enum class AprilTagSampleDirection : uint8_t {
    kLeft = 0,
    kUp = 1,
    kRight = 2,
    kDown = 3,
    kMaxDirectionNumber = 4,
};

struct AprilTag {
    AprilTagFamily famliy = AprilTagFamily::kTag36H11;
    int32_t id = -1;
    std::array<Vec2, 4> vertices;
    std::vector<Vec2> sample_points;
    int32_t hamming_distance = 0;
};

// Shared family metadata and bit layouts, in official codeword order (MSB first).
class AprilTagDefinitions {

public:
    using BitCoordinate = std::array<int32_t, 2>;

    struct FamilyInfo {
        const std::vector<BitCoordinate> *bits = nullptr;
        float border_width = 0.0f;
        const std::vector<uint64_t> *codes = nullptr;  // Only the low bit_num bits are used; white = 1, black = 0.
        int32_t min_valid_hamming_distance = 0;  // Minimum distance between legal codewords, not a correction limit.
        bool reversed_border = false;          // Flexible layouts require the native border geometry (width 1).
    };

    // Covers all 26 families in the official AprilTag and aruco directories.
    // Returns nullptr for an unsupported family; returned data has static lifetime.
    static const FamilyInfo *GetFamilyInfo(const AprilTagFamily &family);
    // Accepts official names (tagAruco6x6_250) and project names (aruco_6x6_250).
    static const AprilTagFamily GetFamilyType(const std::string &family_name_str);

private:
    struct FamilyEntry {
        AprilTagFamily type;
        const char *name;
        const char *official_name;
        const FamilyInfo *info;
    };

    static const std::vector<BitCoordinate> bits_standard_41h12_;
    static const std::vector<uint64_t> codes_standard_41h12_;
    static const FamilyInfo tag_standard_41h12_;
    static const std::vector<BitCoordinate> bits_36h11_;
    static const std::vector<uint64_t> codes_36h11_;
    static const FamilyInfo tag_36h11_;
    static const std::vector<BitCoordinate> bits_aruco_mip_36h12_;
    static const std::vector<uint64_t> codes_aruco_mip_36h12_;
    static const FamilyInfo aruco_mip_36h12_;
    static const std::vector<BitCoordinate> bits_aruco_4x4_100_;
    static const std::vector<uint64_t> codes_aruco_4x4_100_;
    static const FamilyInfo aruco_4x4_100_;
    static const std::vector<BitCoordinate> bits_16h5_;
    static const std::vector<uint64_t> codes_16h5_;
    static const FamilyInfo tag_16h5_;
    static const std::vector<BitCoordinate> bits_25h9_;
    static const std::vector<uint64_t> codes_25h9_;
    static const FamilyInfo tag_25h9_;
    static const std::vector<BitCoordinate> bits_36h10_;
    static const std::vector<uint64_t> codes_36h10_;
    static const FamilyInfo tag_36h10_;
    static const std::vector<BitCoordinate> bits_circle_21h7_;
    static const std::vector<uint64_t> codes_circle_21h7_;
    static const FamilyInfo tag_circle_21h7_;
    static const std::vector<BitCoordinate> bits_circle_49h12_;
    static const std::vector<uint64_t> codes_circle_49h12_;
    static const FamilyInfo tag_circle_49h12_;
    static const std::vector<BitCoordinate> bits_custom_48h12_;
    static const std::vector<uint64_t> codes_custom_48h12_;
    static const FamilyInfo tag_custom_48h12_;
    static const std::vector<BitCoordinate> bits_standard_52h13_;
    static const std::vector<uint64_t> codes_standard_52h13_;
    static const FamilyInfo tag_standard_52h13_;
    static const std::vector<BitCoordinate> bits_aruco_4x4_50_;
    static const std::vector<uint64_t> codes_aruco_4x4_50_;
    static const FamilyInfo aruco_4x4_50_;
    static const std::vector<BitCoordinate> bits_aruco_4x4_250_;
    static const std::vector<uint64_t> codes_aruco_4x4_250_;
    static const FamilyInfo aruco_4x4_250_;
    static const std::vector<BitCoordinate> bits_aruco_4x4_1000_;
    static const std::vector<uint64_t> codes_aruco_4x4_1000_;
    static const FamilyInfo aruco_4x4_1000_;
    static const std::vector<BitCoordinate> bits_aruco_5x5_50_;
    static const std::vector<uint64_t> codes_aruco_5x5_50_;
    static const FamilyInfo aruco_5x5_50_;
    static const std::vector<BitCoordinate> bits_aruco_5x5_100_;
    static const std::vector<uint64_t> codes_aruco_5x5_100_;
    static const FamilyInfo aruco_5x5_100_;
    static const std::vector<BitCoordinate> bits_aruco_5x5_250_;
    static const std::vector<uint64_t> codes_aruco_5x5_250_;
    static const FamilyInfo aruco_5x5_250_;
    static const std::vector<BitCoordinate> bits_aruco_5x5_1000_;
    static const std::vector<uint64_t> codes_aruco_5x5_1000_;
    static const FamilyInfo aruco_5x5_1000_;
    static const std::vector<BitCoordinate> bits_aruco_6x6_50_;
    static const std::vector<uint64_t> codes_aruco_6x6_50_;
    static const FamilyInfo aruco_6x6_50_;
    static const std::vector<BitCoordinate> bits_aruco_6x6_100_;
    static const std::vector<uint64_t> codes_aruco_6x6_100_;
    static const FamilyInfo aruco_6x6_100_;
    static const std::vector<BitCoordinate> bits_aruco_6x6_250_;
    static const std::vector<uint64_t> codes_aruco_6x6_250_;
    static const FamilyInfo aruco_6x6_250_;
    static const std::vector<BitCoordinate> bits_aruco_6x6_1000_;
    static const std::vector<uint64_t> codes_aruco_6x6_1000_;
    static const FamilyInfo aruco_6x6_1000_;
    static const std::vector<BitCoordinate> bits_aruco_7x7_50_;
    static const std::vector<uint64_t> codes_aruco_7x7_50_;
    static const FamilyInfo aruco_7x7_50_;
    static const std::vector<BitCoordinate> bits_aruco_7x7_100_;
    static const std::vector<uint64_t> codes_aruco_7x7_100_;
    static const FamilyInfo aruco_7x7_100_;
    static const std::vector<BitCoordinate> bits_aruco_7x7_250_;
    static const std::vector<uint64_t> codes_aruco_7x7_250_;
    static const FamilyInfo aruco_7x7_250_;
    static const std::vector<BitCoordinate> bits_aruco_7x7_1000_;
    static const std::vector<uint64_t> codes_aruco_7x7_1000_;
    static const FamilyInfo aruco_7x7_1000_;
    static const std::array<FamilyEntry, 26> families_;
};

}  // namespace feature_detector

#endif  // end of _APRIL_TAG_TYPEDEF_H_
