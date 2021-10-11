#include "precomp.hpp"

namespace cv { namespace vsac {
double getLambda (std::vector<int> &supports, double cdf_thr, int points_size,
        int sample_size, bool is_independent, int &min_non_random_inliers) {
    std::sort(supports.begin(), supports.end());
    double lambda = supports.size() % 2 ? (supports[supports.size()/2] + supports[supports.size()/2+1])*0.5 : supports[supports.size()/2];
    const double cdf = lambda + cdf_thr*sqrt(lambda * (1 - lambda / (is_independent ? points_size - sample_size : points_size)));
    int lower_than_cdf = 0; lambda = 0;
    for (const auto &inl : supports)
        if (inl < cdf) {
            lambda += inl; lower_than_cdf++;
        } else break; // list is sorted
    lambda /= lower_than_cdf;
    if (lambda < 1 || lower_than_cdf == 0) lambda = 1;
    // use 0.9999 quantile https://keisan.casio.com/exec/system/14060745333941
    if (! is_independent) // do not calculate it for all inliers
        min_non_random_inliers = (int)(lambda + 3.719*sqrt(lambda * (1 - lambda / points_size))) + 1;
    return lambda;
}

int VSAC::getIndependentInliers (const Mat &model_, const std::vector<int> &sample,
                                 std::vector<int> &inliers_, const int num_inliers_) {
    bool is_F = params.isFundamental();
    Mat model = model_;
    int sample_size;
    if (is_F) sample_size = 7;
    else if (params.isHomography()) sample_size = 4;
    else if (params.isEssential()) {
        is_F = true;
        // convert E to F
        model = Mat(Matx33d(K2).inv().t() * Matx33d(model) * Matx33d(K1).inv());
        sample_size = 5;
    } else if (params.isPnP() || params.getEstimator() == ::vsac::EstimationMethod::Affine) sample_size = 3;
    else assert(false && "Method for independent inliers is not implemented for this problem");
    if (num_inliers_ <= sample_size) return 0; // minimal sample size generates model
    model.convertTo(model, CV_32F);
    int num_inliers = num_inliers_, num_pts_bad_conditioning = 0, num_pts_near_ep = 0, num_pts_on_ep_lines = 0,
            num_pts_validatin_or_constr = 0, pt1 = 0;
    std::vector<int> inliers = inliers_;
    const auto * const pts = params.isEssential() ? (float *) image_points_val.data : (float *) points_val.data;
    // scale for thresholds should be used
    const float ep_thr_sqr = 1e-6, line_thr = .01, neigh_thr = 4;
    float sign1=0,a1=0, b1=0, c1=0, a2=0, b2=0, c2=0, ep1_x, ep1_y, ep2_x, ep2_y;
    const auto * const m = (float *) model.data;
    Vec3f ep1;
    bool do_or_test = false, ep1_inf = false, ep2_inf = false;
    if (is_F) { // compute epipole and sign of the first point for orientation test
        ep1 = Utils::getRightEpipole(model);
        const Vec3f ep2 = Utils::getLeftEpipole(model);
        if (fabsf(ep1[2]) < DBL_EPSILON) {
            ep1_inf = true;
        } else {
            ep1_x = ep1[0] / ep1[2];
            ep1_y = ep1[1] / ep1[2];
        }
        if (fabsf(ep2[2]) < DBL_EPSILON) {
            ep2_inf = true;
        } else {
            ep2_x = ep2[0] / ep2[2];
            ep2_y = ep2[1] / ep2[2];
        }
    }
    const auto * const e1 = ep1.val; // of size 3x1

    // we move sample points to the end, so every inlier will be checked by sample point
    int num_sample_in_inliers = 0;
    if (!sample.empty()) {
        num_sample_in_inliers = 0;
        int temp_idx = num_inliers;
        for (int i = 0; i < temp_idx; i++) {
            const int inl = inliers[i];
            for (int s : sample) {
                if (inl == s) {
                    std::swap(inliers[i], inliers[--temp_idx]);
                    i--; // we need to check inlier that we just swapped
                    num_sample_in_inliers++;
                    break;
                }
            }
        }
    }

    if (is_F) {
        int MIN_TEST = std::min(15, num_inliers);
        for (int i = 0; i < MIN_TEST; i++) {
            pt1 = 4*inliers[i];
            sign1 = (m[0]*pts[pt1+2]+m[3]*pts[pt1+3]+m[6])*(e1[1]-e1[2]*pts[pt1+1]);
            int validate = 0;
            for (int j = 0; j < MIN_TEST; j++) {
                if (i == j) continue;
                const int inl_idx = 4*inliers[j];
                if (sign1*(m[0]*pts[inl_idx+2]+m[3]*pts[inl_idx+3]+m[6])*(e1[1]-e1[2]*pts[inl_idx+1])<0)
                    validate++;
            }
            if (validate < MIN_TEST/2) {
                do_or_test = true; break;
            }
        }
    }

    // verification does not include sample points as they are surely random
    const int max_verify = num_inliers - num_sample_in_inliers;
    if (max_verify <= 0)
        return 0;
    int num_non_random_inliers = num_inliers - sample_size;
    auto removeDependentPoints = [&] (bool do_orient_test, bool check_epipoles) {
        for (int i = 0; i < max_verify; i++) {
            // checks over inliers if they are dependent to other inliers
            const int inl_idx = 4*inliers[i];
            const auto x1 = pts[inl_idx], y1 = pts[inl_idx+1], x2 = pts[inl_idx+2], y2 = pts[inl_idx+3];
            if (is_F) {
                // epipolar line on image 2 = l2
                a2 = m[0] * x1 + m[1] * y1 + m[2];
                b2 = m[3] * x1 + m[4] * y1 + m[5];
                c2 = m[6] * x1 + m[7] * y1 + m[8];
                // epipolar line on image 1 = l1
                a1 = m[0] * x2 + m[3] * y2 + m[6];
                b1 = m[1] * x2 + m[4] * y2 + m[7];
                c1 = m[2] * x2 + m[5] * y2 + m[8];
                if ((!ep1_inf && fabsf(x1-ep1_x)+fabsf(y1-ep1_y) < neigh_thr) ||
                    (!ep2_inf && fabsf(x2-ep2_x)+fabsf(y2-ep2_y) < neigh_thr)) {
                    num_non_random_inliers--;
                    num_pts_near_ep++;
                    continue; // is dependent, continue to the next point
                } else if (check_epipoles) {
                    if (a2 * a2 + b2 * b2 + c2 * c2 < ep_thr_sqr ||
                        a1 * a1 + b1 * b1 + c1 * c1 < ep_thr_sqr) {
                        num_non_random_inliers--;
                        num_pts_near_ep++;
                        continue; // is dependent, continue to the next point
                    }
                }
                else if (do_orient_test && pt1 != inl_idx && sign1*(m[0]*x2+m[3]*y2+m[6])*(e1[1]-e1[2]*y1)<0) {
                    num_non_random_inliers--;
                    num_pts_validatin_or_constr++;
                    continue;
                }
                const auto mag2 = 1 / sqrt(a2 * a2 + b2 * b2), mag1 = 1/sqrt(a1 * a1 + b1 * b1);
                a2 *= mag2; b2 *= mag2; c2 *= mag2;
                a1 *= mag1; b1 *= mag1; c1 *= mag1;
            }

            for (int j = i+1; j < num_inliers; j++) {// verify through all including sample points
                const int inl_idx_j = 4*inliers[j];
                const auto X1 = pts[inl_idx_j], Y1 = pts[inl_idx_j+1], X2 = pts[inl_idx_j+2], Y2 = pts[inl_idx_j+3];
//                const double dx1 = X1-x1, dy1 = Y1-y1, dx2 = X2-x2, dy2 = Y2-y2;
//                if (dx1 * dx1 + dy1 * dy1 < neigh_thr_sqr || dx2 * dx2 + dy2 * dy2 < neigh_thr_sqr) {
                // use L1 norm instead of L2 for faster evaluation
                if (fabsf(X1-x1) + fabsf(Y1 - y1) < neigh_thr || fabsf(X2-x2) + fabsf(Y2 - y2) < neigh_thr) {
                    num_non_random_inliers--;
                    // num_pts_bad_conditioning++;
                    break; // is dependent stop verification
                } else if (is_F) {
                    if (fabsf(a2 * X2 + b2 * Y2 + c2) < line_thr && //|| // xj'^T F   xi
                        fabsf(a1 * X1 + b1 * Y1 + c1) < line_thr) { // xj^T  F^T xi'
                        num_non_random_inliers--;
                        // num_pts_on_ep_lines++;
                        break; // is dependent stop verification
                    }
                }
            }
        }
    };
    if (params.isPnP()) {
        for (int i = 0; i < max_verify; i++) {
            const int inl_idx = 5*inliers[i];
            const auto x = pts[inl_idx], y = pts[inl_idx+1], X = pts[inl_idx+2], Y = pts[inl_idx+3], Z = pts[inl_idx+4];
            for (int j = i+1; j < num_inliers; j++) {
                const int inl_idx_j = 5*inliers[j];
                if (fabsf(x-pts[inl_idx_j  ]) + fabsf(y-pts[inl_idx_j+1]) < neigh_thr ||
                    fabsf(X-pts[inl_idx_j+2]) + fabsf(Y-pts[inl_idx_j+3]) + fabsf(Z-pts[inl_idx_j+4]) < neigh_thr) {
                    num_non_random_inliers--;
                    break;
                }
            }
        }
    } else {
        removeDependentPoints(do_or_test, !ep1_inf && !ep2_inf);
        if (is_F) {
            const bool is_pts_vald_constr_normal = (double)num_pts_validatin_or_constr / num_inliers < 0.6;
            const bool is_pts_near_ep_normal = (double)num_pts_near_ep / num_inliers < 0.6;
            if (!is_pts_near_ep_normal || !is_pts_vald_constr_normal) {
                num_non_random_inliers = num_inliers-sample_size;
                num_pts_bad_conditioning = 0; num_pts_near_ep = 0; num_pts_on_ep_lines = 0; num_pts_validatin_or_constr = 0;
                removeDependentPoints(is_pts_vald_constr_normal, is_pts_near_ep_normal);
            }
        }
    }
    return num_non_random_inliers;
}
}}