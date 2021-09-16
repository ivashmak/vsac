#include "precomp.hpp"

//#define DEBUG_DEGENSAC true
#define DEBUG_DEGENSAC false

namespace cv { namespace vsac {
class EpipolarGeometryDegeneracyImpl : public EpipolarGeometryDegeneracy {
private:
    const Mat * points_mat;
    const float * const points; // i-th row xi1 yi1 xi2 yi2
    const int min_sample_size;
public:
    explicit EpipolarGeometryDegeneracyImpl (const Mat &points_, int sample_size_) :
        points_mat(&points_), points ((float*) points_.data), min_sample_size (sample_size_) {}
    /*
     * Do oriented constraint to verify if epipolar geometry is in front or behind the camera.
     * Return: true if all points are in front of the camers w.r.t. tested epipolar geometry - satisfies constraint.
     *         false - otherwise.
     * x'^T F x = 0
     * e' × x' ~+ Fx   <=>  λe' × x' = Fx, λ > 0
     * e  × x ~+ x'^T F
     */
    inline bool isModelValid(const Mat &F_, const std::vector<int> &sample) const override {
        // F is of rank 2, taking cross product of two rows we obtain null vector of F
        Vec3d ec_mat = F_.row(0).cross(F_.row(2));
        auto * ec = ec_mat.val; // of size 3x1

        // e is zero vector, recompute e
        if (ec[0] <= 1.9984e-15 && ec[0] >= -1.9984e-15 &&
            ec[1] <= 1.9984e-15 && ec[1] >= -1.9984e-15 &&
            ec[2] <= 1.9984e-15 && ec[2] >= -1.9984e-15) {
            ec_mat = F_.row(1).cross(F_.row(2));
            ec = ec_mat.val;
        }
        const auto * const F = (double *) F_.data;

        // without loss of generality, let the first point in sample be in front of the camera.
        int pt = 4*sample[0];
        // check only two first elements of vectors (e × x) and (x'^T F)
        // s1 = (x'^T F)[0] = x2 * F11 + y2 * F21 + 1 * F31
        // s2 = (e × x)[0] = e'_2 * 1 - e'_3 * y1
        // sign1 = s1 * s2
        const double sign1 = (F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(ec[1]-ec[2]*points[pt+1]);

        for (int i = 1; i < min_sample_size; i++) {
            pt = 4 * sample[i];
            // if signum of the first point and tested point differs
            // then two points are on different sides of the camera.
            if (sign1*(F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(ec[1]-ec[2]*points[pt+1])<0)
                    return false;
        }
        return true;
    }
    void getEpipoles (const cv::Mat &F, Vec3d &ep1, double &ep1_x, double &ep1_y, double &ep2_x, double &ep2_y) const {
        ep1 = F.row(0).cross(F.row(2));
        auto * e = ep1.val;

        // e is zero vector, recompute e
        if (e[0] <= 1.9984e-15 && e[0] >= -1.9984e-15 &&
            e[1] <= 1.9984e-15 && e[1] >= -1.9984e-15 &&
            e[2] <= 1.9984e-15 && e[2] >= -1.9984e-15) {
            ep1 = F.row(1).cross(F.row(2));
        }

        cv::Vec3d ep2 = F.col(0).cross(F.col(2));
        e = ep2.val;

        // e is zero vector, recompute e
        if (e[0] <= 1.9984e-15 && e[0] >= -1.9984e-15 &&
            e[1] <= 1.9984e-15 && e[1] >= -1.9984e-15 &&
            e[2] <= 1.9984e-15 && e[2] >= -1.9984e-15) {
            ep2 = F.col(1).cross(F.col(2));
        }

        // std::cout << "test ep1 " << cv::norm(F * ep1) << " ep2 " << cv::norm(F.t() * ep2) << '\n';
        if (fabs(ep1[2]) < DBL_EPSILON) {
            ep1_x = DBL_MAX;
            ep1_y = DBL_MAX;
        } else {
            ep1_x = ep1[0] / ep1[2];
            ep1_y = ep1[1] / ep1[2];
        }
        if (fabs(ep2[2]) < DBL_EPSILON) {
            ep2_x = DBL_MAX;
            ep2_y = DBL_MAX;
        } else {
            ep2_x = ep2[0] / ep2[2];
            ep2_y = ep2[1] / ep2[2];
        }
    }

    void filterInliers (const cv::Mat &F, std::vector<bool> &inliers_mask) const override {
        std::vector<int> inliers(inliers_mask.size());
        int num_inliers = 0, pt = 0;
        for (bool is_inlier : inliers_mask) {
            if (is_inlier)
                inliers[num_inliers++] = pt;
            pt++;
        }
        const int num_new_inliers = filterInliers(F, inliers, num_inliers);
        std::fill(inliers_mask.begin(), inliers_mask.end(), false);
        for (int i = 0; i < num_new_inliers; i++)
            inliers_mask[inliers[i]] = true;
    }
    int filterInliers (const cv::Mat &F, std::vector<int> &inliers, int num_inliers) const override {
        const auto * const m = (double *) F.data;
        cv::Vec3d ep1;
        double ep1_x, ep1_y, ep2_x, ep2_y;
        getEpipoles(F, ep1, ep1_x, ep1_y, ep2_x, ep2_y);
        auto * e1 = ep1.val;

        int anchor_pt = 0;
        bool anchor_pt_found = false;
        const int max_pts_test = std::min(30, (int)inliers.size());
        for (int p = 0; p < num_inliers; p++) {
            anchor_pt = 4*inliers[p];
            const int pt1 = 4*anchor_pt;
            const double sign1 = (m[0]*points[pt1+2]+m[3]*points[pt1+3]+m[6])*(e1[1]-e1[2]*points[pt1+1]);
            int num_incorrect_pts = 0;
            for (int i = 0; i < max_pts_test; i++) {
                const int inl_idx = 4*inliers[i];
                if (pt1 != inl_idx && sign1*(m[0]*points[inl_idx+2]+m[3]*points[inl_idx+3]+m[6])*(e1[1]-e1[2]*points[inl_idx+1])<0)
                    num_incorrect_pts++;
            }
            if ((double)num_incorrect_pts / max_pts_test < 0.2) {
                anchor_pt_found = true;
                break;
            }
        }

        const int pt1 = 4*anchor_pt;
        const double sign1 = (m[0]*points[pt1+2]+m[3]*points[pt1+3]+m[6])*(e1[1]-e1[2]*points[pt1+1]);
        const auto sgd_error = cv::vsac::SymmetricGeometricDistance::create(*points_mat);
        sgd_error->setModelParameters(F);
        int num_new_inliers = num_inliers;
        const double ep_thr_sqr = 50, double_sgd_thr_sqr = 30;
        for (int inl = 0; inl < num_inliers; inl++) {
            const int inl_idx = 4 * inliers[inl];
            const double x1 = points[inl_idx], y1 = points[inl_idx+1], x2 = points[inl_idx+2], y2 = points[inl_idx+3];

            if (anchor_pt_found && pt1 != inl_idx && sign1*(m[0]*x2+m[3]*y2+m[6])*(e1[1]-e1[2]*y1)<0) {
                std::swap(inliers[inl], inliers[--num_new_inliers]);
                continue;
            }

            if (pow(ep1_x - x1, 2) + pow(ep1_y - y1, 2) < ep_thr_sqr ||
                pow(ep2_x - x2, 2) + pow(ep2_y - y2, 2) < ep_thr_sqr) {
                std::swap(inliers[inl], inliers[--num_new_inliers]);
                continue;
            }

            if (sgd_error->getError(inliers[inl]) > double_sgd_thr_sqr) {
                std::swap(inliers[inl], inliers[--num_new_inliers]);
                continue;
            }
        }
        return num_new_inliers;
    }
};
void EpipolarGeometryDegeneracy::recoverRank (Mat &model, bool is_fundamental_mat) {
    /*
     * Do singular value decomposition.
     * Make last eigen value zero of diagonal matrix of singular values.
     */
    Matx33d U, Vt;
    Vec3d w;
    SVD::compute(model, w, U, Vt, SVD::MODIFY_A);
    if (is_fundamental_mat)
        model = Mat(U * Matx33d(w(0), 0, 0, 0, w(1), 0, 0, 0, 0) * Vt);
    else {
        const double mean_singular_val = (w[0] + w[1]) * 0.5;
        model = Mat(U * Matx33d(mean_singular_val, 0, 0, 0, mean_singular_val, 0, 0, 0, 0) * Vt);
    }
}
Ptr<EpipolarGeometryDegeneracy> EpipolarGeometryDegeneracy::create (const Mat &points_,
        int sample_size_) {
    return makePtr<EpipolarGeometryDegeneracyImpl>(points_, sample_size_);
}

class HomographyDegeneracyImpl : public HomographyDegeneracy {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit HomographyDegeneracyImpl (const Mat &points_) :
            points_mat(&points_), points ((float *)points_.data) {}

    inline bool isSampleGood (const std::vector<int> &sample) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2], smpl4 = 4*sample[3];
        // planar correspondences must lie on the same side of any line from two points in sample
        const float x1 = points[smpl1], y1 = points[smpl1+1], X1 = points[smpl1+2], Y1 = points[smpl1+3];
        const float x2 = points[smpl2], y2 = points[smpl2+1], X2 = points[smpl2+2], Y2 = points[smpl2+3];
        const float x3 = points[smpl3], y3 = points[smpl3+1], X3 = points[smpl3+2], Y3 = points[smpl3+3];
        const float x4 = points[smpl4], y4 = points[smpl4+1], X4 = points[smpl4+2], Y4 = points[smpl4+3];
        // line from points 1 and 2
        const float ab_cross_x = y1 - y2, ab_cross_y = x2 - x1, ab_cross_z = x1 * y2 - y1 * x2;
        const float AB_cross_x = Y1 - Y2, AB_cross_y = X2 - X1, AB_cross_z = X1 * Y2 - Y1 * X2;

        // check if points 3 and 4 are on the same side of line ab on both images
        if ((ab_cross_x * x3 + ab_cross_y * y3 + ab_cross_z) *
            (AB_cross_x * X3 + AB_cross_y * Y3 + AB_cross_z) < 0)
            return false;
        if ((ab_cross_x * x4 + ab_cross_y * y4 + ab_cross_z) *
            (AB_cross_x * X4 + AB_cross_y * Y4 + AB_cross_z) < 0)
            return false;

        // line from points 3 and 4
        const float cd_cross_x = y3 - y4, cd_cross_y = x4 - x3, cd_cross_z = x3 * y4 - y3 * x4;
        const float CD_cross_x = Y3 - Y4, CD_cross_y = X4 - X3, CD_cross_z = X3 * Y4 - Y3 * X4;

        // check if points 1 and 2 are on the same side of line cd on both images
        if ((cd_cross_x * x1 + cd_cross_y * y1 + cd_cross_z) *
            (CD_cross_x * X1 + CD_cross_y * Y1 + CD_cross_z) < 0)
            return false;
        if ((cd_cross_x * x2 + cd_cross_y * y2 + cd_cross_z) *
            (CD_cross_x * X2 + CD_cross_y * Y2 + CD_cross_z) < 0)
            return false;

        // Checks if points are not collinear
        // If area of triangle constructed with 3 points is less then threshold then points are collinear:
        //           |x1 y1 1|             |x1      y1      1|
        // (1/2) det |x2 y2 1| = (1/2) det |x2-x1   y2-y1   0| = (1/2) det |x2-x1   y2-y1| < threshold
        //           |x3 y3 1|             |x3-x1   y3-y1   0|             |x3-x1   y3-y1|
        // for points on the first image
        if (fabsf((x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)) * 0.5 < FLT_EPSILON) return false; //1,2,3
        if (fabsf((x2-x1) * (y4-y1) - (y2-y1) * (x4-x1)) * 0.5 < FLT_EPSILON) return false; //1,2,4
        if (fabsf((x3-x1) * (y4-y1) - (y3-y1) * (x4-x1)) * 0.5 < FLT_EPSILON) return false; //1,3,4
        if (fabsf((x3-x2) * (y4-y2) - (y3-y2) * (x4-x2)) * 0.5 < FLT_EPSILON) return false; //2,3,4
        // for points on the second image
        if (fabsf((X2-X1) * (Y3-Y1) - (Y2-Y1) * (X3-X1)) * 0.5 < FLT_EPSILON) return false; //1,2,3
        if (fabsf((X2-X1) * (Y4-Y1) - (Y2-Y1) * (X4-X1)) * 0.5 < FLT_EPSILON) return false; //1,2,4
        if (fabsf((X3-X1) * (Y4-Y1) - (Y3-Y1) * (X4-X1)) * 0.5 < FLT_EPSILON) return false; //1,3,4
        if (fabsf((X3-X2) * (Y4-Y2) - (Y3-Y2) * (X4-X2)) * 0.5 < FLT_EPSILON) return false; //2,3,4

        return true;
    }
};
Ptr<HomographyDegeneracy> HomographyDegeneracy::create (const Mat &points_) {
    return makePtr<HomographyDegeneracyImpl>(points_);
}

class FundamentalDegeneracyImpl : public FundamentalDegeneracy {
private:
    RNG rng;
    const Ptr<Quality> quality;
    const Ptr<Error> f_error;
    Ptr<Quality> h_repr_quality;
    const float * const points;
    const Mat * points_mat;
    const Ptr<ReprojectionErrorForward> h_reproj_error;
    Ptr<HomographyNonMinimalSolver> h_non_min_solver;
    const EpipolarGeometryDegeneracyImpl ep_deg;
    // threshold to find inliers for homography model
    const double homography_threshold, log_conf = log(0.05);
    double likely_homogr_thr, f_threshold_sqr;
    // points (1-7) to verify in sample
    std::vector<std::vector<int>> h_sample {{0,1,2},{3,4,5},{0,1,6},{3,4,6},{2,5,6}};
    std::vector<int> h_inliers, good_sample, h_outliers, new_h_outliers;
    std::vector<double> weights;
    std::vector<Mat> h_models;
    const int points_size, sample_size, max_iters_plane_and_parallax, TENT_MIN_NON_PLANAR_SUPP = 10;
    const int MAX_MODELS_TO_TEST = 12, H_INLS_DEGEN_SAMPLE = 4;
    std::vector<int> non_planar_supports;
    // int first_sample_pt = 0;
    // re-estimate for every H
    int num_h_outliers, num_models_used_so_far = 0, estimated_min_non_planar_support = TENT_MIN_NON_PLANAR_SUPP;
    Matx33d K, K2, K_inv, K2_inv, K2_inv_t, true_K2_inv, true_K2_inv_t, true_K1_inv, true_K1;
    Mat H_best;
    Score H_best_score;
    bool true_K_given;
    std::vector<std::vector<int>> close_pts_mask;

    // best estimated pose
    cv::Mat R_est_best, K_est_best, t_est_best;
public:
    FundamentalDegeneracyImpl (int state, const Ptr<Quality> &quality_, const Mat &points_,
                int sample_size_, int plane_and_parallax_iters, double homography_threshold_,
                double f_inlier_thr_sqr, const Mat true_K1_, const Mat true_K2_) :
            rng (state), quality(quality_), f_error(quality_->getErrorFnc()), points((float *) points_.data), points_mat(&points_),
            h_reproj_error(ReprojectionErrorForward::create(points_)),
            ep_deg (points_, sample_size_), homography_threshold (homography_threshold_),
            points_size (quality_->getPointsSize()), sample_size (sample_size_),
            max_iters_plane_and_parallax(plane_and_parallax_iters) {
        if (sample_size_ == 8) {
            // add more homography samples to test for 8-points F
            h_sample.emplace_back(std::vector<int>{0, 1, 7});
            h_sample.emplace_back(std::vector<int>{0, 2, 7});
            h_sample.emplace_back(std::vector<int>{3, 5, 7});
            h_sample.emplace_back(std::vector<int>{3, 6, 7});
            h_sample.emplace_back(std::vector<int>{2, 4, 7});
        }
        non_planar_supports = std::vector<int>(MAX_MODELS_TO_TEST);
        h_inliers = std::vector<int>(points_size);
        h_outliers = std::vector<int>(points_size);
        new_h_outliers = std::vector<int>(points_size);
        h_non_min_solver = HomographyNonMinimalSolver::create(points_);
        likely_homogr_thr = 100; // 10^2
        num_h_outliers = points_size;
        // non_planar_support_of_F_from_K = 0;
        f_threshold_sqr = f_inlier_thr_sqr;
        // f_threshold_sqr = quality_->getThreshold();
        H_best_score = Score();
        h_repr_quality = MsacQuality::create(points_.rows, homography_threshold_, h_reproj_error);
        true_K_given = ! true_K1_.empty();
        if (true_K_given) {
            const Mat K2_inv_ = true_K2_.inv();
            const auto * const k1 = (double *) true_K1_.data, * const k2_inv = (double *) K2_inv_.data;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    true_K1(i,j) = k1[i*3+j];
                    true_K2_inv(i,j) = k2_inv[i*3+j];
                }
            true_K1_inv = true_K1.inv();
            true_K2_inv_t = true_K2_inv.t();            
        }
    }
    inline bool isModelValid(const Mat &F, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(F, sample);
    }
    void setPrincipalPoint (double px_, double py_) override {
        setPrincipalPoint(px_, py_, 0, 0);
    }
    void setPrincipalPoint (double px_, double py_, double px2_, double py2_) override {
        K = {1, 0, px_, 0, 1, py_, 0, 0, 1};
        if (px2_ != 0)
            K2 = {1, 0, px2_, 0, 1, py2_, 0, 0, 1};
        else K2 = K;
        K_inv = Matx33d::eye();
        K2_inv = Matx33d::eye();
        K2_inv_t = Matx33d::eye();
    }
    bool estimateHfrom3Points (const Mat &F_best, const std::vector<int> &sample) {
#if DEBUG_DEGENSAC
        const auto h_est_time = std::chrono::steady_clock::now();
#endif
        H_best_score = Score(); H_best = Mat();

        // find e', null space of F^T
        Vec3d e_prime = F_best.col(0).cross(F_best.col(2));
        if (fabs(e_prime(0)) < 1e-10 && fabs(e_prime(1)) < 1e-10 &&
            fabs(e_prime(2)) < 1e-10) // if e' is zero
            e_prime = F_best.col(1).cross(F_best.col(2));

        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);

        Vec3d xi_prime(0,0,1), xi(0,0,1), b;
        Matx33d M(0,0,1,0,0,1,0,0,1); // last column of M is 1

        bool is_degenerate = false;
        for (const auto &h_i : h_sample) { // only 5 samples
            for (int pt_i = 0; pt_i < 3; pt_i++) {
                // find b and M
                const int smpl = 4*sample[h_i[pt_i]];
                xi[0] = points[smpl];
                xi[1] = points[smpl+1];
                xi_prime[0] = points[smpl+2];
                xi_prime[1] = points[smpl+3];

                // (x′i × e')
                const Vec3d xprime_X_eprime = xi_prime.cross(e_prime);

                // (x′i × (A xi))
                const Vec3d xprime_X_Ax = xi_prime.cross(A * xi);

                // x′i × (A xi))^T (x′i × e′) / ‖x′i×e′‖^2,
                b[pt_i] = xprime_X_Ax.dot(xprime_X_eprime) /
                          std::pow(norm(xprime_X_eprime), 2);

                // M from x^T
                M(pt_i, 0) = xi[0];
                M(pt_i, 1) = xi[1];
            }

            // compute H
            Matx33d H = A - e_prime * (M.inv() * b).t();
            h_reproj_error->setModelParameters(Mat(H));
            int inliers_in_plane = 0;
            for (int s = 0; s < sample_size; s++)
                if (h_reproj_error->getError(sample[s]) < likely_homogr_thr)
                    if (++inliers_in_plane >= H_INLS_DEGEN_SAMPLE)
                        break;

            // dangerous if happen that 4 points on plane and all other 3 points are not true correspondonces
            if (inliers_in_plane >= H_INLS_DEGEN_SAMPLE) { // checks H for non-randomness
                is_degenerate = true;
                const auto h_score = h_repr_quality->getScore(Mat(H));
                if (h_score.isBetter(H_best_score)) {
                    H_best_score = h_score;
                    H_best = Mat(H);
                }
            }
        }
#if DEBUG
        std::cout << "H est time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - h_est_time).count() << '\n';
#endif
        if (!is_degenerate)
            return false;

#if DEBUG_DEGENSAC
        const auto h_opt_time = std::chrono::steady_clock::now();
#endif
        int h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
        const int max_iters = 6, max_pts_H = 50;
        int max_h_sample = std::min(max_pts_H, (int)(0.8*h_inls_cnt));
        if (h_inls_cnt > max_h_sample && max_h_sample >= 4/*min H sample size*/) {
            auto rand_gen = UniformRandomGenerator::create(0, points_size, max_h_sample);
            for (int iter = 0; iter < max_iters; iter++) {
                if (h_non_min_solver->estimate(rand_gen->generateUniqueRandomSubset(h_inliers, h_inls_cnt), max_h_sample, h_models, weights) == 0)
                    continue;
                const auto h_score = h_repr_quality->getScore(h_models[0]);
                if (h_score.isBetter(H_best_score)) {
#if DEBUG_DEGENSAC
                    // std::cout << "H SCORE UPDATE LO at " << iter << " (" << H_best_score.score << ", " << H_best_score.inlier_number << ") -> (" << h_score.score << ", " << h_score.inlier_number << ")\n";
#endif
                    H_best_score = h_score;
                    h_models[0].copyTo(H_best);
                    h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
                    const int new_max_h_sample = std::min(max_pts_H, (int)(0.8*h_inls_cnt));
                    if (new_max_h_sample > max_h_sample) {
                        max_h_sample = new_max_h_sample;
                        rand_gen = UniformRandomGenerator::create(max_h_sample/*state*/, points_size, max_h_sample);
                    }
                }
            }
        }
        for (int iter = 0; iter < 2; iter++) {
            if (h_non_min_solver->estimate(h_inliers, h_inls_cnt, h_models, weights) == 0)
                break;
            const auto h_score = h_repr_quality->getScore(h_models[0]);
            if (h_score.isBetter(H_best_score)) {
#if DEBUG_DEGENSAC
                // std::cout << "H SCORE UPDATE FO at " << iter << " (" << H_best_score.score << ", " << H_best_score.inlier_number << ") -> (" << h_score.score << ", " << h_score.inlier_number << ")\n";
#endif
                H_best_score = h_score;
                h_models[0].copyTo(H_best);
                h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
            } else break;
        }
        h_reproj_error->setModelParameters(H_best);
#if DEBUG_DEGENSAC
        std::cout << "H optimization time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - h_opt_time).count() << '\n';
#endif
        // re-estimate avg non-planar support
        num_models_used_so_far = 0; num_h_outliers = 0;
        estimated_min_non_planar_support = TENT_MIN_NON_PLANAR_SUPP;
        const auto &h_errors = h_reproj_error->getErrors(H_best);
        for (int pt = 0; pt < points_size; pt++)
            if (h_errors[pt] > likely_homogr_thr)
                h_outliers[num_h_outliers++] = pt;

#if DEBUG_DEGENSAC
        const auto filter_time = std::chrono::steady_clock::now();
#endif
        filterHoutliers();
#if DEBUG_DEGENSAC
//        std::cout << "filter time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - filter_time).count() << '\n';
#endif
        ////////////////////////////////////////////

#if DEBUG_DEGENSAC
//        std::cout << "H SCORE (" << H_best_score.score << ", " << H_best_score.inlier_number << " H outliers " << num_h_outliers << " / " << points_size << ")\n";
#endif
        return true;
    }
    void filterHoutliers () {
        /////// remove close points from H outliers
        std::random_shuffle(&h_outliers[0], &h_outliers[0] + num_h_outliers);
        int new_num_h_outliers = 0;
        std::vector<bool> unique(points_size, true);
        if (close_pts_mask.empty()) {
#if DEBUG_DEGENSAC
            std::cout << "compute close points\n";
#endif
            Utils::getClosePoints(*points_mat, close_pts_mask, 5.25);
        }
        for (int i = 0; i < num_h_outliers; i++) {
            if (!unique[i]) continue;
            for (int n : close_pts_mask[h_outliers[i]])
                unique[n] = false;
        }
        for (int i = 0; i < num_h_outliers; i++)
            if (unique[h_outliers[i]])
                new_h_outliers[new_num_h_outliers++] = h_outliers[i];
        h_outliers = new_h_outliers;
        num_h_outliers = new_num_h_outliers;
#if DEBUG_DEGENSAC
        std::cout << "num h outliers " << num_h_outliers << " new " << new_num_h_outliers << '\n';
#endif
    }
    void setClosePointsMask (const std::vector<std::vector<int>> &close_pts) override { close_pts_mask = close_pts; }
    bool recoverIfDegenerate (const std::vector<int> &sample, const Mat &F_best, const Score &F_best_score,
                              Mat &non_degenerate_model, Score &non_degenerate_model_score, int &non_planar_support) override {
        const auto swapF = [&] (const Mat &_F, int _support, const Score &_score) {
            _F.copyTo(non_degenerate_model); non_planar_support = _support; non_degenerate_model_score = _score;
        };
        if (! estimateHfrom3Points(F_best, sample)) {
            non_planar_support = -1; // so far H does not exist but maybe appear in the future
#if DEBUG_DEGENSAC
            std::cout << "H does not exist, SO-FAR-THE-BEST F IS NOT DEGENERATE\n";
#endif
            return false; // non degenerate
        }
        cv::Mat F_from_K;
        Score F_from_K_score;
        if (true_K_given) {
            if (!getFfromTrueK(H_best, F_from_K, F_from_K_score)) {
                non_degenerate_model_score = Score();
                return true; // no translation
            }
        }
#if DEBUG_DEGENSAC
        std::cout << "RETURN F FROM GIVEN K1, K2 (" << F_best_score.score << ", " << F_best_score.inlier_number << ") -> (" << score_f2.score << ", " << score_f2.inlier_number << ")\n";
#endif

//        int num_f_inliers_h_outliers;
        const int F_best_non_planar_support = getNonPlanarSupport(F_best);//, num_f_inliers_h_outliers);
        if (!isFDegenerate(F_best_non_planar_support)) {
#if DEBUG_DEGENSAC
            std::cout << "SO-FAR-THE-BEST F SCORE (" << F_best_score.score << ", " << F_best_score.inlier_number << ") IS NOT DEGENERATE BY NON-RAND TEST, support  " << F_best_non_planar_support << "\n";
#endif
            if (F_from_K_score.isBetter(F_best_score)) {
                swapF(F_from_K, -1, F_from_K_score);
                return true;
            }
            return false;
        }
#if DEBUG_DEGENSAC
        std::cout << "SFTB F IS DEGEN, NON-PLANAR SUPPORT " << F_best_non_planar_support << " SCORE (" << F_best_score.score << ", " << F_best_score.inlier_number << ") H INLIERS " << H_best_score.inlier_number << " H OUTLIERS " << num_h_outliers << "\n";
#endif
        if (true_K_given) {
            swapF(F_from_K, -1, F_from_K_score);
            return true;
        }

#if DEBUG_DEGENSAC
        const auto calib_deg_time = std::chrono::steady_clock::now();
#endif
        int F_calib_non_planar_support;
        Score F_calib_score;
        Mat F_calib = calibDegensac(H_best, F_calib_score, F_calib_non_planar_support);
        if (!F_calib.empty()) {
            if (!isFDegenerate(F_calib_non_planar_support)) {
                    swapF(F_calib, F_calib_non_planar_support, F_calib_score);
#if DEBUG_DEGENSAC
                std::cout << "Calib-deg time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -calib_deg_time).count() << '\n';
                std::cout << "RETURN F FROM APPRX K1, K2 (" << F_best_score.score << ", " << F_best_score.inlier_number << ") -> (" << F_calib_score.score << ", " << F_calib_score.inlier_number << ")\n";
#endif
                return true;
            }
        }
#if DEBUG_DEGENSAC
        std::cout << "Calib-deg time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -calib_deg_time).count() <<
            ", score (" << F_calib_score.score << ", " << F_calib_score.inlier_number << "), support " << F_calib_non_planar_support<< '\n';
#endif

#if DEBUG_DEGENSAC
        const auto parallax_time = std::chrono::steady_clock::now();
#endif
        int F_pl_par_non_planar_support;
        Score F_pl_par_score;
        Mat F_pl_par = planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, max_iters_plane_and_parallax, true, F_pl_par_score, F_pl_par_non_planar_support);
        if (!F_pl_par.empty()) {
            if (!isFDegenerate(F_pl_par_non_planar_support)) {
                swapF(F_pl_par, F_pl_par_non_planar_support, F_pl_par_score);
#if DEBUG_DEGENSAC
                std::cout << "Plane-&-parallax time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - parallax_time).count() << '\n';
                std::cout << "RETURN F FROM PL-&-PAR (" << F_best_score.score << ", " << F_best_score.inlier_number << ") -> (" << F_pl_par_score.score << ", " << F_pl_par_score.inlier_number << ")\n";
#endif
                return true;
            }        
        }
#if DEBUG_DEGENSAC
        std::cout << "Plane-&-parallax time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - parallax_time).count()
            << ", score (" << F_pl_par_score.score << ", " << F_pl_par_score.inlier_number << "), support " << F_pl_par_non_planar_support << '\n';
#endif

        non_degenerate_model_score = Score();
        non_planar_support = F_best_non_planar_support;
#if DEBUG_DEGENSAC
        std::cout << "FAILED TO RECOVER, best " << F_best_non_planar_support << " calib " <<
            F_calib_non_planar_support << " par " << F_pl_par_non_planar_support << " est " << estimated_min_non_planar_support << " (rough " << estimated_min_non_planar_support_rough << ") " <<
        "; #inls " << F_best_score.inlier_number << " calib " << F_calib_score.inlier_number << " " << " " << F_pl_par_score.inlier_number << "\n";
#endif
        return true;
    }

    const std::vector<int> &getGoodSample() override { return good_sample; }
    const Mat &getHomography () const override { return H_best; }
    const std::vector<int> &getNonPlanarPoints () const override { return h_outliers; }
    // RANSAC with plane-and-parallax to find new Fundamental matrix
    bool getFfromTrueK (const Matx33d &H, Mat &F_from_K, Score &F_from_K_score) {
        std::vector<Matx33d> R;
        std::vector<Vec3d> t;
        const int num_sols = Utils::decomposeHomography(true_K2_inv * H * true_K1, R, t);
        if (num_sols == 1) {
            // std::cout << "Warning: translation is zero!\n";
            return false; // is degenerate
        }
        // sign of translation does not make difference
        const Mat F1 = Mat(true_K2_inv_t * Math::getSkewSymmetric(t[0]) * R[0] * true_K1_inv);
        const Mat F2 = Mat(true_K2_inv_t * Math::getSkewSymmetric(t[1]) * R[1] * true_K1_inv);
        const auto score_f1 = quality->getScore(F1);
        const auto score_f2 = quality->getScore(F2);
        if (score_f1.isBetter(score_f2)) {
            F_from_K = F1;
            F_from_K_score = score_f1;
        } else {
            F_from_K = F2;
            F_from_K_score = score_f2;
        }
        return true;
    }
    Mat planeAndParallaxRANSAC (const Matx33d &H, const std::vector<int> &non_planar_pts, int num_non_planar_pts,
            int max_iters_pl_par, bool use_preemptive, Score &H_out_score, int &non_planar_support_out) {
        if (num_non_planar_pts < 2) {
            non_planar_support_out = 0;
            return Mat();
        }
        num_models_used_so_far = 0; // reset estimation of lambda for plane-and-parallax
        int max_iters = max_iters_pl_par, max_inliers_out_of_h = 0, iters = 0;
        std::vector<Matx33d> possibly_good_models;
        for (; iters < max_iters; iters++) {
            // draw two random points
            int h_outlier1 = non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            int h_outlier2 = non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            while (h_outlier1 == h_outlier2)
                h_outlier2 = non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            // do plane and parallax with outliers of H
            // F = [(p1' x Hp1) x (p2' x Hp2)]_x H
            const Matx33d F = Math::getSkewSymmetric(
                    (Vec3d(points[4*h_outlier1+2], points[4*h_outlier1+3], 1).cross   // p1'
               (H * Vec3d(points[4*h_outlier1  ], points[4*h_outlier1+1], 1))).cross // Hp1
                     (Vec3d(points[4*h_outlier2+2], points[4*h_outlier2+3], 1).cross   // p2'
               (H * Vec3d(points[4*h_outlier2  ], points[4*h_outlier2+1], 1)))       // Hp2
            ) * H;

            const int num_f_inliers_of_h_outliers = getNonPlanarSupport(Mat(F));
            // std::cout << "pl & par, non planar support " << num_f_inliers_of_h_outliers << " score " << quality->getScore(Mat(F)).score << "\n"; 
            if (max_inliers_out_of_h < num_f_inliers_of_h_outliers) {
                max_inliers_out_of_h = num_f_inliers_of_h_outliers;
                const double predicted_iters = log_conf / log(1 - std::pow
                        (static_cast<double>(num_f_inliers_of_h_outliers) / num_h_outliers, 2));
                if (use_preemptive && ! std::isinf(predicted_iters) && predicted_iters < max_iters)
                    max_iters = static_cast<int>(predicted_iters);
                possibly_good_models = { F };
            } else if (max_inliers_out_of_h == num_f_inliers_of_h_outliers)
                possibly_good_models.emplace_back(F);
        }

        non_planar_support_out = max_inliers_out_of_h;
        Mat best_F;
        H_out_score = Score();
        int idx = 0;
        for (const auto &F_ : possibly_good_models) {
            const auto sc = quality->getScore(Mat(F_));
            if (sc.isBetter(H_out_score)) {
                H_out_score = sc;
                best_F = Mat(F_);
            }
            idx++;
        }
#if DEBUG_DEGENSAC
        std::cout << "plane-and-parallax " << H_out_score.score << " " << H_out_score.inlier_number << " out " << non_planar_support_out << " iters " << iters << "\n";
#endif
        return best_F;
    }
    void getBestKRt (Mat &K_, Mat &R_, Mat &t_) const override {
        K_est_best.copyTo(K_); R_est_best.copyTo(R_); t_est_best.copyTo(t_);
    }
    cv::Mat calibDegensac (const Matx33d &H, Score &H_out_score, int &non_planar_support_out) {
        if (K(0,2) == 0 && K(1,2) == 0) return Mat();
        num_models_used_so_far = 0; // reset estimation of lambda for calib degensac
        std::vector<Matx33d> R;
        std::vector<Vec3d> t;
        std::vector<Mat> possibly_good_models;
        non_planar_support_out = 0;

        for (double f = 300; f <= 3500; f += 150.) {
            K(0,0) = K(1,1) = K2(0,0) = K2(1,1) = f;
            const double one_over_f = 1/f;
            K_inv(0,0) = K_inv(1,1) = K2_inv(0,0) = K2_inv(1,1) = K2_inv_t(0,0) = K2_inv_t(1,1) = one_over_f;
            K_inv(0,2) = -K(0,2)*one_over_f; K_inv(1,2) = -K(1,2)*one_over_f;
            K2_inv_t(2,0) = K2_inv(0,2) = -K2(0,2)*one_over_f; K2_inv_t(2,1) = K2_inv(1,2) = -K2(1,2)*one_over_f;

            const int sols = Utils::decomposeHomography(K2_inv * H * K, R, t);
            if (sols == 1) continue;
            const Mat F1 = Mat(K2_inv_t * Math::getSkewSymmetric(t[0]) * R[0] * K_inv);
            const Mat F2 = Mat(K2_inv_t * Math::getSkewSymmetric(t[1]) * R[1] * K_inv);
            const int non_planar_f1 = getNonPlanarSupport(F1), non_planar_f2 = getNonPlanarSupport(F2);
//            std::cout << "f " << f << " non planar supports " << non_planar_f1 << " " << non_planar_f2 << " best " << non_planar_support_out <<
//                        " inliers " << quality->getScore(F1).inlier_number << " "<< quality->getScore(F2).inlier_number
//                        << " scores " << quality->getScore(F1).score << " " << quality->getScore(F2).score << "\n";
            if (non_planar_f1 > non_planar_f2) {
                if (non_planar_support_out < non_planar_f1) {
                    non_planar_support_out = non_planar_f1;
                    possibly_good_models = {F1};
                } else if (non_planar_support_out == non_planar_f1) {
                    possibly_good_models.emplace_back(F1);
                }
            } else {
                if (non_planar_support_out < non_planar_f2) {
                    non_planar_support_out = non_planar_f2;
                    possibly_good_models = {F2};
                } else if (non_planar_support_out == non_planar_f2) {
                    possibly_good_models.emplace_back(F2);
                }
            }
        }

        /*
        // logarithmic search -> faster but less accurate
        double f_min = 300, f_max = 3500;
        while (f_max - f_min > 100) {
            const double f_half = (f_max + f_min) * 0.5f, left_half = (f_min + f_half) * 0.5f, right_half = (f_half + f_max) * 0.5f;
            const double inl_in_left = eval_f(left_half), inl_in_right = eval_f(right_half);
            if (inl_in_left > inl_in_right)
                f_max = f_half;
            else f_min = f_half;
        }
        */

        Mat best_F;
        H_out_score = Score();
        int idx = 0;
        double best_f = 0;
        for (const auto &F_ : possibly_good_models) {
            const auto sc = quality->getScore(F_);
            if (sc.isBetter(H_out_score)) {
                H_out_score = sc;
                F_.copyTo(best_F);
            }
            idx++;
        }

#if DEBUG_DEGENSAC
    std::cout << "calib degensac " << H_out_score.score << " " << H_out_score.inlier_number << " out " << non_planar_support_out << "\n";
#endif
        return best_F;
    }

    bool verifyFundamental (const Mat &F_best, const Score &F_score, const std::vector<bool> &inliers_mask, cv::Mat &F_new, Score &new_score) override {
        const int max_H_iters = 4; // 3.52 = log(0.01) / log(1 - std::pow(0.9, 3));
        int num_f_inliers = 0;
        std::vector<int> inliers(points_size);
        for (int i = 0; i < points_size; i++)
            if (inliers_mask[i])
                inliers[num_f_inliers++] = i;
        const int f_sample_size = 3;
        const auto sampler = UniformSampler::create(0, f_sample_size, num_f_inliers);
        std::vector<int> f_sample(f_sample_size);
        // find e', null space of F^T
        Vec3d e_prime = F_best.col(0).cross(F_best.col(2));
        if (fabs(e_prime(0)) < 1e-10 && fabs(e_prime(1)) < 1e-10 &&
            fabs(e_prime(2)) < 1e-10) // if e' is zero
            e_prime = F_best.col(1).cross(F_best.col(2));
        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);

        Vec3d xi_prime(0,0,1), xi(0,0,1), b;
        Matx33d M(0,0,1,0,0,1,0,0,1); // last column of M is 1

        H_best_score = Score();
        H_best = Mat();
        for (int iter = 0; iter < max_H_iters; iter++) {
            sampler->generateSample(f_sample);

            for (int pt_i = 0; pt_i < f_sample_size; pt_i++) {
                // find b and M
                const int smpl = 4*inliers[f_sample[pt_i]];
                xi[0] = points[smpl];
                xi[1] = points[smpl+1];
                xi_prime[0] = points[smpl+2];
                xi_prime[1] = points[smpl+3];

                // (x′i × e')
                const Vec3d xprime_X_eprime = xi_prime.cross(e_prime);

                // (x′i × (A xi))
                const Vec3d xprime_X_Ax = xi_prime.cross(A * xi);

                // x′i × (A xi))^T (x′i × e′) / ‖x′i×e′‖^2,
                b[pt_i] = xprime_X_Ax.dot(xprime_X_eprime) /
                          std::pow(norm(xprime_X_eprime), 2);

                // M from x^T
                M(pt_i, 0) = xi[0];
                M(pt_i, 1) = xi[1];
            }

            // compute H
            Matx33d H = A - e_prime * (M.inv() * b).t();
            const auto h_score = h_repr_quality->getScore(Mat(H));
            if (h_score.isBetter(H_best_score)) {
                H_best_score = h_score;
                H_best = Mat(H);
            }
        }
        // get H outliers
        num_h_outliers = 0;
        const auto &errors = h_reproj_error->getErrors(Mat(H_best));
        for (int i = 0; i < points_size; i++)
            if (errors[i] > likely_homogr_thr)
                h_outliers[num_h_outliers++] = i;
        filterHoutliers();

        const int F_support = getNonPlanarSupport(F_best);
        const bool is_F_degen = isFDegenerate(F_support);
#if DEBUG_DEGENSAC
        std::cout << "F final verification: num h outliers " << num_h_outliers << " F non planar support " << F_support << " pts size " << points_size << "\n";
#endif

        // save new F, if F from K is better than so-far-the-best
        if (true_K_given) {
            // get non-degenerate F from K
            Mat F_from_K;
            Score F_from_K_score;
            if (!getFfromTrueK(H_best, F_from_K, F_from_K_score)) {
                if (F_from_K_score.isBetter(F_score) || is_F_degen) {
                    F_from_K.copyTo(F_new);
                    new_score = F_from_K_score;
                    return true;
                }
            }
        } else {
            // still run calibrated DEGENSAC to obtain non-degenerate F
            int F_calib_non_planar_support;
            Score F_calib_score;
            Mat F_calib = calibDegensac(H_best, F_calib_score, F_calib_non_planar_support);
            if (F_calib_score.isBetter(F_score) || is_F_degen) {
                F_calib.copyTo(F_new);
                new_score = F_calib_score;
                return true;
            }
        }

        if (!is_F_degen) {
#if DEBUG_DEGENSAC
        std::cout << "final F is not degenerate\n";
#endif
            return false;
        }
        // so-far-the-best F is degenerate:
        Score plane_parallax_score;
        int plane_parallax_support;
        // run fast plane-and-parallax to recover it
        cv::Mat F_plane_parallax = planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, 20, true, plane_parallax_score, plane_parallax_support);
        if (!F_plane_parallax.empty() && !isFDegenerate(plane_parallax_support)) {
            new_score = plane_parallax_score;
            F_plane_parallax.copyTo(F_new);
            return true;
        }
        // plane-and-parallax failed. A previous non-degenerate so-far-the-best model will be used instead
        new_score = Score();
        return true;
    }
private:
    int getNonPlanarSupport (const Mat &F) {
        int non_rand_support = 0;
        f_error->setModelParameters(F);
        for (int pt = 0; pt < num_h_outliers; pt++)
            if (f_error->getError(h_outliers[pt]) < f_threshold_sqr)
                non_rand_support++;
        if (num_models_used_so_far < MAX_MODELS_TO_TEST && !true_K_given/*for K we know that recovered F cannot be degenerate*/) {
            non_planar_supports[num_models_used_so_far++] = non_rand_support;
            if (num_models_used_so_far == MAX_MODELS_TO_TEST) {
                std::sort(non_planar_supports.begin(), non_planar_supports.end());
#if DEBUG_DEGENSAC
//                for (const auto v : non_planar_supports) std::cout << v << " ";
//                std::cout << "\n";
#endif
                const int med_support = non_planar_supports[MAX_MODELS_TO_TEST/2];
                double perc99 = med_support + 2.32 * sqrt(med_support * (1 - (double)med_support/num_h_outliers));
                double avg_support = 0, num_below_perc = 0;
                for (auto v : non_planar_supports) {
                    if (v > perc99)
                        break;
                    avg_support += v;
                    num_below_perc++;
                }
                estimated_min_non_planar_support = std::max(3 /*at least 3 points out of plane*/, (int)(avg_support/num_below_perc)+1);
#if DEBUG_DEGENSAC
                std::cout << "ESTIMATED MIN NON-RAND SUPP, rough " << estimated_min_non_planar_support_rough << " est " <<
                    estimated_min_non_planar_support << " med " << med_support << " delta " << (double)med_support/num_h_outliers
                    << " 99% " << perc99 << " avg supp " << avg_support / num_below_perc << "\n";
#endif
            }
        }
        return non_rand_support;
    }
    bool isFDegenerate (int num_f_inliers_h_outliers) const {
        if (num_models_used_so_far < MAX_MODELS_TO_TEST)
            // the minimum number of non-planar support has not estimated yet -> use tentative
            return num_f_inliers_h_outliers < std::min(TENT_MIN_NON_PLANAR_SUPP, (int)(0.5 * num_h_outliers));
        return num_f_inliers_h_outliers < estimated_min_non_planar_support;   
    }
};
Ptr<FundamentalDegeneracy> FundamentalDegeneracy::create (int state, const Ptr<Quality> &quality_,
        const Mat &points_, int sample_size_, int max_iters_plane_and_parallax, double homography_threshold_,
        double f_inlier_thr_sqr, const Mat true_K1, const Mat true_K2) {
    return makePtr<FundamentalDegeneracyImpl>(state, quality_, points_, sample_size_,
              max_iters_plane_and_parallax, homography_threshold_, f_inlier_thr_sqr, true_K1, true_K2);
}

class EssentialDegeneracyImpl : public EssentialDegeneracy {
private:
    const Mat * points_mat;
    const int sample_size;
    const EpipolarGeometryDegeneracyImpl ep_deg;
public:
    explicit EssentialDegeneracyImpl (const Mat &points, int sample_size_) :
            points_mat(&points), sample_size(sample_size_), ep_deg (points, sample_size_) {}
    inline bool isModelValid(const Mat &E, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(E, sample);
    }
};
Ptr<EssentialDegeneracy> EssentialDegeneracy::create (const Mat &points_, int sample_size_) {
    return makePtr<EssentialDegeneracyImpl>(points_, sample_size_);
}
}}
