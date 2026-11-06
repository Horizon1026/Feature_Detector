#include "feature_point_detector.h"

#include "quadratic_plane.h"
#include "slam_basic_math.h"
#include "slam_operations.h"

namespace feature_detector {

/* Class Feature Point Detector Definition. */
bool FeaturePointDetector::DetectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    // Check input image.
    RETURN_FALSE_IF(image.data() == nullptr);

    // If there are already some detected features, do not detect new features besiding them.
    if (features.empty()) {
        mask_.setConstant(image.rows(), image.cols(), 1);
    } else {
        UpdateMaskByFeatures(image, features);
    }

    // Detect all features to be candidates.
    candidates_.clear();
    RETURN_FALSE_IF_FALSE(ComputeCandidates(image));

    // Select good features by score from candidates.
    RETURN_FALSE_IF_FALSE(SelectGoodFeatures(image, needed_feature_num, features));
    return true;
}

void FeaturePointDetector::SparsifyFeatures(const std::vector<Vec2> &features, const int32_t image_rows, const int32_t image_cols,
                                            const uint8_t status_need_filter, const uint8_t status_after_filter, std::vector<uint8_t> &status) {
    if (features.size() != status.size()) {
        status.assign(features.size(), 1);
    }

    // Grid filter to make points sparsely.
    const float grid_row_step = image_rows / (options_.kGridFilterRowDivideNumber - 1);
    const float grid_col_step = image_cols / (options_.kGridFilterColDivideNumber - 1);
    mask_.setConstant(options_.kGridFilterRowDivideNumber, options_.kGridFilterColDivideNumber, 1);
    for (uint32_t i = 0; i < features.size(); ++i) {
        const int32_t row = static_cast<int32_t>(features[i].y() / grid_row_step);
        const int32_t col = static_cast<int32_t>(features[i].x() / grid_col_step);

        if (row < 0 || row > mask_.rows() - 1 || col < 0 || col > mask_.cols() - 1) {
            status[i] = status_after_filter;
            continue;
        }

        if (mask_(row, col) && status[i] == status_need_filter) {
            mask_(row, col) = 0;
        } else if (!mask_(row, col) && status[i] == status_need_filter) {
            status[i] = status_after_filter;
        }
    }
}

bool FeaturePointDetector::SelectGoodFeatures(const GrayImage &image, const uint32_t needed_feature_num, std::vector<Vec2> &features) {
    RETURN_TRUE_IF(candidates_.empty());

    // Sort candidates by response in descending order.
    std::sort(candidates_.begin(), candidates_.end(), [](const std::pair<float, Pixel> &a, const std::pair<float, Pixel> &b) {
        return a.first > b.first;
    });

    for (const auto &candidate: candidates_) {
        const Pixel pixel = candidate.second;
        const int32_t row = pixel.y();
        const int32_t col = pixel.x();
        if (mask_(row, col)) {
            features.emplace_back(Vec2(pixel.x(), pixel.y()));
            RETURN_TRUE_IF(features.size() >= needed_feature_num);
            DrawRectangleInMask(row, col);
        }
    }

    return true;
}

void FeaturePointDetector::DrawRectangleInMask(const int32_t row, const int32_t col) {
    for (int32_t drow = -options_.kMinFeatureDistance; drow <= options_.kMinFeatureDistance; ++drow) {
        for (int32_t dcol = -options_.kMinFeatureDistance; dcol <= options_.kMinFeatureDistance; ++dcol) {
            const int32_t sub_row = drow + row;
            const int32_t sub_col = dcol + col;
            if (sub_row < 0 || sub_col < 0 || sub_row > mask_.rows() - 1 || sub_col > mask_.cols() - 1) {
                continue;
            }

            mask_(sub_row, sub_col) = 0;
        }
    }
}

void FeaturePointDetector::UpdateMaskByFeatures(const GrayImage &image, const std::vector<Vec2> &features) {
    mask_.setConstant(image.rows(), image.cols(), 1);

    for (const auto &feature: features) {
        const int32_t row = feature.y();
        const int32_t col = feature.x();
        DrawRectangleInMask(row, col);
    }
}

bool FeaturePointDetector::RefineFeatureToSubPixelPrecisionByParabolic(const GrayImage &image, const Vec2 &raw_feature, Vec2 &refined_feature) {
    RETURN_FALSE_IF(image.data() == nullptr);

    const Pixel mid_pixel = raw_feature.cast<int32_t>();
    std::vector<Vec3> samples;
    const int32_t radius = options_.sub_pixel.kRadiusForSubPixelRefine;
    samples.reserve(radius * radius * 4);
    for (int32_t dx = -radius; dx <= radius; ++dx) {
        for (int32_t dy = -radius; dy <= radius; ++dy) {
            uint8_t value = 0;
            if (image.GetPixelValue(mid_pixel.y() + dy, mid_pixel.x() + dx, &value)) {
                samples.emplace_back(Vec3(dx, dy, static_cast<float>(value)));
            }
        }
    }

    QuadraticPlane quadratic_plane;
    RETURN_FALSE_IF(!quadratic_plane.FitParabolicModel(samples));
    Vec3 saddle_point = Vec3::Zero();
    RETURN_FALSE_IF(!quadratic_plane.ComputeSaddlePoint(saddle_point));
    RETURN_FALSE_IF(saddle_point.head<2>().norm() > options_.sub_pixel.kMaxToleranceSubPixelRefineStep);

    QuadraticPlane::Curvatures curvatures;
    RETURN_FALSE_IF(!quadratic_plane.ComputeCurvaturesAtPoint(saddle_point, curvatures));

    const float k1 = std::fabs(curvatures.max_principal_curvature);
    const float k2 = std::fabs(curvatures.min_principal_curvature);
    RETURN_FALSE_IF(!std::isfinite(k1) || !std::isfinite(k2));
    RETURN_FALSE_IF(!(curvatures.gaussian_curvature < 0.0f));

    const float min_k = std::min(k1, k2);
    const float max_k = std::max(k1, k2);
    RETURN_FALSE_IF(min_k < options_.sub_pixel.parabolic.kMinTolarenceCurvatureForSubPixelRefine);
    RETURN_FALSE_IF(min_k / max_k < options_.sub_pixel.parabolic.kMinToleranceCurvatureRatioForSubPixelRefine);

    const Vec2 result = mid_pixel.cast<float>() + saddle_point.head<2>();
    RETURN_FALSE_IF(result.x() < 0 || result.x() > image.cols() - 1 || result.y() < 0 || result.y() > image.rows() - 1);
    refined_feature = result;
    return true;
}

bool FeaturePointDetector::RefineFeatureToSubPixelPrecisionByGradient(const GrayImage &image, const Vec2 &raw_feature, Vec2 &refined_feature) {
    RETURN_FALSE_IF(image.data() == nullptr);
    RETURN_FALSE_IF(!raw_feature.allFinite());

    // Reserve one pixel for central differences and another for bilinear interpolation.
    const int32_t radius = options_.sub_pixel.kRadiusForSubPixelRefine;
    const float border = static_cast<float>(radius) + 1.0f;
    RETURN_FALSE_IF(raw_feature.x() < border || raw_feature.x() > image.cols() - 2.0f - border || raw_feature.y() < border ||
                    raw_feature.y() > image.rows() - 2.0f - border);

    // The Gaussian window is separable and remains fixed during refinement.
    std::vector<float> weights(2 * radius + 1);
    for (int32_t i = -radius; i <= radius; ++i) {
        const float distance = static_cast<float>(i) / radius;
        weights[i + radius] = std::exp(-distance * distance);
    }

    Vec2 result = raw_feature;
    for (int32_t iter = 0; iter < options_.sub_pixel.gradient.kMaxIteration; ++iter) {
        RETURN_FALSE_IF(result.x() < border || result.x() > image.cols() - 2.0f - border || result.y() < border || result.y() > image.rows() - 2.0f - border);

        // Minimize sum(weight * (gradient.transpose() * (step - offset))^2).
        float gxx = 0.0;
        float gxy = 0.0;
        float gyy = 0.0;
        float bx = 0.0;
        float by = 0.0;
        for (int32_t dy = -radius; dy <= radius; ++dy) {
            for (int32_t dx = -radius; dx <= radius; ++dx) {
                const float row = result.y() + dy;
                const float col = result.x() + dx;
                const float gx = image.GetPixelValueNoCheck(row, col + 1.0f) - image.GetPixelValueNoCheck(row, col - 1.0f);
                const float gy = image.GetPixelValueNoCheck(row + 1.0f, col) - image.GetPixelValueNoCheck(row - 1.0f, col);
                const float weight = weights[dy + radius] * weights[dx + radius];
                const float xx = weight * gx * gx;
                const float xy = weight * gx * gy;
                const float yy = weight * gy * gy;
                gxx += xx;
                gxy += xy;
                gyy += yy;
                bx += xx * dx + xy * dy;
                by += xy * dx + yy * dy;
            }
        }

        // Reject flat patches and nearly singular systems caused by a single edge.
        const float trace = gxx + gyy;
        const float determinant = gxx * gyy - gxy * gxy;
        RETURN_FALSE_IF(trace <= 0.0 || determinant <= kZeroFloat * trace * trace);

        const Vec2 step((gyy * bx - gxy * by) / determinant, (gxx * by - gxy * bx) / determinant);
        RETURN_FALSE_IF(!step.allFinite());
        result += step;
        RETURN_FALSE_IF(!result.allFinite());
        RETURN_FALSE_IF((result - raw_feature).norm() > options_.sub_pixel.kMaxToleranceSubPixelRefineStep);
        RETURN_FALSE_IF(result.x() < 0 || result.x() > image.cols() - 1 || result.y() < 0 || result.y() > image.rows() - 1);

        if (step.squaredNorm() <= options_.sub_pixel.gradient.kConvergeStepNorm * options_.sub_pixel.gradient.kConvergeStepNorm) {
            refined_feature = result;
            return true;
        }
    }

    // Do not accept a point which has not converged within the iteration limit.
    return false;
}

bool FeaturePointDetector::RefineFeatureToSubPixelPrecisionBySymmetry(const GrayImage &image, const Vec2 &raw_feature, Vec2 &refined_feature) {
    const auto &config = options_.sub_pixel.symmetry;
    const int32_t radius = options_.sub_pixel.kRadiusForSubPixelRefine;
    const float max_shift = options_.sub_pixel.kMaxToleranceSubPixelRefineStep;
    RETURN_FALSE_IF(image.data() == nullptr || !raw_feature.allFinite());
    RETURN_FALSE_IF(radius < 1 || image.rows() < 2 || image.cols() < 2);
    RETURN_FALSE_IF(radius > (std::min(image.rows(), image.cols()) - 2) / 2);

    // Keep the whole (2r + 1)^2 patch and its bilinear interpolation footprint in the image.
    const auto IsInBounds = [&](const Vec2 &center) {
        return center.allFinite() && center.x() >= radius && center.y() >= radius && center.x() <= image.cols() - 2.0f - radius &&
               center.y() <= image.rows() - 2.0f - radius;
    };
    RETURN_FALSE_IF(!IsInBounds(raw_feature));

    // Differentiate the same bilinear interpolant used for the photometric residual.
    const auto Sample = [&](float x, float y, Vec2 &gradient) {
        const int32_t col = static_cast<int32_t>(x);
        const int32_t row = static_cast<int32_t>(y);
        const float fx = x - col;
        const float fy = y - row;
        const float a = image.GetPixelValueNoCheck(row, col);
        const float b = image.GetPixelValueNoCheck(row, col + 1);
        const float c = image.GetPixelValueNoCheck(row + 1, col);
        const float d = image.GetPixelValueNoCheck(row + 1, col + 1);
        gradient = Vec2((1.0f - fy) * (b - a) + fy * (d - c), (1.0f - fx) * (c - a) + fx * (d - b));
        return (1.0f - fy) * ((1.0f - fx) * a + fx * b) + fy * ((1.0f - fx) * c + fx * d);
    };
    const float side = 2.0f * radius + 1.0f;
    const float count = side * side - 1.0f;
    const auto evaluate = [&](const Vec2 &center, Mat2 &hessian, Vec2 &rhs) {
        hessian.setZero();
        rhs.setZero();
        float cost = 0.0f;
        // Use each opposite pair once; the center has identically zero residual and Jacobian.
        for (int32_t dy = 0; dy <= radius; ++dy) {
            for (int32_t dx = -radius; dx <= radius; ++dx) {
                if (dy == 0 && dx <= 0) {
                    continue;
                }
                Vec2 positive_gradient, negative_gradient;
                const float positive = Sample(center.x() + dx, center.y() + dy, positive_gradient);
                const float negative = Sample(center.x() - dx, center.y() - dy, negative_gradient);
                const float residual = positive - negative;
                // Both samples move with c: d[I(c+d)-I(c-d)]/dc = grad I(c+d)-grad I(c-d).
                const Vec2 jacobian = positive_gradient - negative_gradient;
                hessian.noalias() += jacobian * jacobian.transpose();
                rhs.noalias() += jacobian * residual;
                cost += residual * residual;
            }
        }
        hessian *= 2.0f / count;
        rhs *= 2.0f / count;
        return cost * 2.0f / count;
    };

    Vec2 result = raw_feature;
    for (int32_t iter = 0; iter < config.kMaxIteration; ++iter) {
        Mat2 hessian;
        Vec2 rhs;
        const float cost = evaluate(result, hessian, rhs);
        const float trace = hessian.trace();
        const float discriminant = std::hypot(hessian(0, 0) - hessian(1, 1), 2.0f * hessian(0, 1));
        const float max_eigenvalue = 0.5f * (trace + discriminant);
        const float min_eigenvalue = 0.5f * (trace - discriminant);
        // Flat patches and single edges cannot determine a unique 2D center.
        RETURN_FALSE_IF(min_eigenvalue < config.kMinEigenvalue || min_eigenvalue < config.kMinEigenvalueRatio * max_eigenvalue);
        const Vec2 step = -hessian.ldlt().solve(rhs);
        RETURN_FALSE_IF(!step.allFinite());
        if (step.norm() <= config.kConvergeStepNorm) {
            RETURN_FALSE_IF(cost > config.kMaxRmsPhotometricError * config.kMaxRmsPhotometricError);
            refined_feature = result;
            return true;
        }

        // Backtracking protects against overshooting narrow, blurred edges. A clipped step is never treated as convergence; the next iteration must satisfy the
        // full GN step.
        bool accepted = false;
        float scale = 1.0f;
        for (int32_t trial = 0; trial < 16; ++trial, scale *= 0.5f) {
            const Vec2 candidate = result + scale * step;
            if (!IsInBounds(candidate) || (candidate - raw_feature).norm() > max_shift) {
                continue;
            }
            Mat2 unused_hessian;
            Vec2 unused_rhs;
            if (evaluate(candidate, unused_hessian, unused_rhs) < cost) {
                result = candidate;
                accepted = true;
                break;
            }
        }
        RETURN_FALSE_IF(!accepted);
    }
    return false;
}

}  // namespace feature_detector
