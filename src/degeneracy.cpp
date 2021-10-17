#include "precomp.hpp"

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
        const Vec3d ep = Utils::getRightEpipole(F_);
        const auto * const e = ep.val; // of size 3x1
        const auto * const F = (double *) F_.data;

        // without loss of generality, let the first point in sample be in front of the camera.
        int pt = 4*sample[0];
        // check only two first elements of vectors (e × x) and (x'^T F)
        // s1 = (x'^T F)[0] = x2 * F11 + y2 * F21 + 1 * F31
        // s2 = (e × x)[0] = e'_2 * 1 - e'_3 * y1
        // sign1 = s1 * s2
        const double sign1 = (F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(e[1]-e[2]*points[pt+1]);

        for (int i = 1; i < min_sample_size; i++) {
            pt = 4 * sample[i];
            // if signum of the first point and tested point differs
            // then two points are on different sides of the camera.
            if (sign1*(F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(e[1]-e[2]*points[pt+1])<0)
                    return false;
        }
        return true;
    }
    void getEpipoles (const Mat &F, Vec3d &ep1, double &ep1_x, double &ep1_y, double &ep2_x, double &ep2_y) const {
        ep1 = Utils::getRightEpipole(F);
        const Vec3d ep2 = Utils::getLeftEpipole(F);
        // std::cout << "test ep1 " << norm(F * ep1) << " ep2 " << norm(F.t() * ep2) << '\n';
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

    void filterInliers (const Mat &F, std::vector<bool> &inliers_mask) const override {
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
    int filterInliers (const Mat &F, std::vector<int> &inliers, int num_inliers) const override {
        const auto * const m = (double *) F.data;
        Vec3d ep1;
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
        const auto sgd_error = vsac::SymmetricGeometricDistance::create(*points_mat);
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
    const float TOLERANCE = 2 * FLT_EPSILON; // 2 from area of triangle
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
        // (1/2) det |x2 y2 1| = (1/2) det |x2-x1   y2-y1   0| = det |x2-x1   y2-y1| < 2 * threshold
        //           |x3 y3 1|             |x3-x1   y3-y1   0|       |x3-x1   y3-y1|
        // for points on the first image
        if (fabsf((x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)) < TOLERANCE) return false; //1,2,3
        if (fabsf((x2-x1) * (y4-y1) - (y2-y1) * (x4-x1)) < TOLERANCE) return false; //1,2,4
        if (fabsf((x3-x1) * (y4-y1) - (y3-y1) * (x4-x1)) < TOLERANCE) return false; //1,3,4
        if (fabsf((x3-x2) * (y4-y2) - (y3-y2) * (x4-x2)) < TOLERANCE) return false; //2,3,4
        // for points on the second image
        if (fabsf((X2-X1) * (Y3-Y1) - (Y2-Y1) * (X3-X1)) < TOLERANCE) return false; //1,2,3
        if (fabsf((X2-X1) * (Y4-Y1) - (Y2-Y1) * (X4-X1)) < TOLERANCE) return false; //1,2,4
        if (fabsf((X3-X1) * (Y4-Y1) - (Y3-Y1) * (X4-X1)) < TOLERANCE) return false; //1,3,4
        if (fabsf((X3-X2) * (Y4-Y2) - (Y3-Y2) * (X4-X2)) < TOLERANCE) return false; //2,3,4

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
    Ptr<UniformRandomGenerator> random_gen_H;
    const EpipolarGeometryDegeneracyImpl ep_deg;
    // threshold to find inliers for homography model
    const double homography_threshold, log_conf = log(0.05), MAX_H_THR = 225/*15^2*/;
    double f_threshold_sqr, best_focal = -1;
    // points (1-7) to verify in sample
    std::vector<std::vector<int>> h_sample {{0,1,2},{3,4,5},{0,1,6},{3,4,6},{2,5,6}};
    std::vector<int> h_inliers, h_outliers, new_h_outliers, plane_parallax_H_sample;
    std::vector<double> weights;
    std::vector<Mat> h_models;
    const int points_size, sample_size, max_iters_plane_and_parallax, TENT_MIN_NON_PLANAR_SUPP = 10, MAX_H_SUBSET = 50, MAX_ITERS_H = 6;
    const int MAX_MODELS_TO_TEST = 21, H_INLS_DEGEN_SAMPLE = 5; // 5 by DEGENSAC, Chum et al.
    std::vector<int> non_planar_supports;
    // int first_sample_pt = 0;
    // re-estimate for every H
    int num_h_outliers, num_models_used_so_far = 0, estimated_min_non_planar_support = TENT_MIN_NON_PLANAR_SUPP,
        plane_parallax_H_sample_size;
    Matx33d K, K2, K_inv, K2_inv, K2_inv_t, true_K2_inv, true_K2_inv_t, true_K1_inv, true_K1, true_K2_t;
    Mat H_best;
    Score H_best_score, best_focal_score;
    bool true_K_given, is_principal_pt_set = false;
    std::vector<std::vector<int>> close_pts_mask;
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
        plane_parallax_H_sample = std::vector<int>(points_size);
        h_non_min_solver = HomographyNonMinimalSolver::create(points_);
        num_h_outliers = points_size;
        f_threshold_sqr = f_inlier_thr_sqr;
        H_best_score = Score();
        h_repr_quality = MsacQuality::create(points_.rows, homography_threshold_, h_reproj_error);
        true_K_given = ! true_K1_.empty() && ! true_K2_.empty();
        if (true_K_given) {
            true_K1 = Matx33d((double *)true_K1_.data);
            true_K2_inv = Matx33d(Mat(true_K2_.inv()));
            true_K2_t = Matx33d(true_K2_).t();
            true_K1_inv = true_K1.inv();
            true_K2_inv_t = true_K2_inv.t();            
        }
        random_gen_H = UniformRandomGenerator::create(rng.uniform(0, INT_MAX), points_size, MAX_H_SUBSET);
        plane_parallax_H_sample_size = 0;
    }
    bool estimateHfrom3Points (const Mat &F_best, const std::vector<int> &sample) {
#ifdef DEBUG_DEGENSAC
        const auto h_est_time = std::chrono::steady_clock::now();
#endif
        H_best_score = Score(); H_best = Mat();
        // find e', null vector of F^T
        const Vec3d e_prime = Utils::getLeftEpipole(F_best);
        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);
        bool is_degenerate = false;
        for (const auto &h_i : h_sample) { // only 5 samples
            Matx33d H;
            if (!getH(A, e_prime, 4*sample[h_i[0]], 4*sample[h_i[1]], 4*sample[h_i[2]], H))
                continue;
            h_reproj_error->setModelParameters(Mat(H));
            int inliers_in_plane = 0;
            for (int s = 0; s < sample_size; s++)
                if (h_reproj_error->getError(sample[s]) < homography_threshold)
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
#ifdef DEBUG_DEGENSAC
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - h_est_time).count() << " H est time, #inls " << H_best_score.inlier_number << "\n";
#endif
        if (!is_degenerate)
            return false;

#ifdef DEBUG_DEGENSAC
        const auto h_opt_time = std::chrono::steady_clock::now();
#endif
        int h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
        random_gen_H->setSubsetSize(h_inls_cnt <= MAX_H_SUBSET ? (int)(0.8*h_inls_cnt) : MAX_H_SUBSET);
        if (random_gen_H->getSubsetSize() >= 4/*min H sample size*/) {
            for (int iter = 0; iter < MAX_ITERS_H; iter++) {
                if (h_non_min_solver->estimate(random_gen_H->generateUniqueRandomSubset(h_inliers, h_inls_cnt), random_gen_H->getSubsetSize(), h_models, weights) == 0)
                    continue;
                const auto h_score = h_repr_quality->getScore(h_models[0]);
                if (h_score.isBetter(H_best_score)) {
#ifdef DEBUG_DEGENSAC
                    // std::cout << "H SCORE UPDATE LO at " << iter << " (" << H_best_score.score << ", " << H_best_score.inlier_number << ") -> (" << h_score.score << ", " << h_score.inlier_number << ")\n";
#endif
                    h_models[0].copyTo(H_best);
                    // if more inliers than previous best
                    if (h_score.inlier_number > H_best_score.inlier_number || h_score.inlier_number >= MAX_H_SUBSET) {
                        h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
                        random_gen_H->setSubsetSize(h_inls_cnt <= MAX_H_SUBSET ? (int)(0.8*h_inls_cnt) : MAX_H_SUBSET);
                    }
                    H_best_score = h_score;
                }
            }
        }
        for (int iter = 0; iter < 2; iter++) {
            if (h_non_min_solver->estimate(h_inliers, h_inls_cnt, h_models, weights) == 0)
                break;
            const auto h_score = h_repr_quality->getScore(h_models[0]);
            if (h_score.isBetter(H_best_score)) {
#ifdef DEBUG_DEGENSAC
                // std::cout << "H SCORE UPDATE FO at " << iter << " (" << H_best_score.score << ", " << H_best_score.inlier_number << ") -> (" << h_score.score << ", " << h_score.inlier_number << ")\n";
#endif
                H_best_score = h_score;
                h_models[0].copyTo(H_best);
                h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
            } else break;
        }
#ifdef DEBUG_DEGENSAC
        std::cout << "H optimization time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - h_opt_time).count() << '\n';
#endif
        getOutliersH(F_best);
        return true;
    }
    bool recoverIfDegenerate (const std::vector<int> &sample, const Mat &F_best, const Score &F_best_score,
                              Mat &non_degenerate_model, Score &non_degenerate_model_score) override {
        if (true_K_given && areSameSingularValues(F_best)) return false;
//#ifdef DEBUG_DEGENSAC
//        std::cout << "- - - - - - - - - - - START DEGENSAC #inls " << F_best_score.inlier_number << " - - - - - - - - - - - - - -\n";
//#endif
        const auto swapF = [&] (const Mat &_F, const Score &_score) {
            _F.copyTo(non_degenerate_model); non_degenerate_model_score = _score;
        };
        if (! estimateHfrom3Points(F_best, sample)) {
            // so far H does not exist but maybe appear in the future
#ifdef DEBUG_DEGENSAC
            std::cout << "H does not exist, SO-FAR-THE-BEST F IS NOT DEGENERATE\n";
#endif
            return false; // non degenerate
        }
        Mat F_from_K; Score F_from_K_score;
        if (true_K_given && !getFfromTrueK(H_best, F_from_K, F_from_K_score)) {
            non_degenerate_model_score = Score();
            return true; // no translation
        }
#ifdef DEBUG_DEGENSAC
        std::cout << "SFTB F IS DEGEN, NON-PLANAR SUPPORT " << 0 << " SCORE (" << F_best_score.score << ", " << F_best_score.inlier_number << ") H INLIERS " << H_best_score.inlier_number << " H OUTLIERS " << num_h_outliers << "\n";
#endif
        if (true_K_given) {
            swapF(F_from_K, F_from_K_score);
            return true;
        }
        Score F_calib_score; Mat F_calib;
        if (calibDegensac(H_best, F_calib, F_calib_score, F_best, F_best_score)) {
            swapF(F_calib, F_calib_score);
            return true;
        }

        Score F_pl_par_score; Mat F_pl_par;
        if (planeAndParallaxRANSAC(H_best, plane_parallax_H_sample, plane_parallax_H_sample_size, max_iters_plane_and_parallax, true, F_pl_par, F_pl_par_score) ||
            planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, max_iters_plane_and_parallax, true, F_pl_par, F_pl_par_score)) {
            swapF(F_pl_par, F_pl_par_score);
//#ifdef DEBUG_DEGENSAC
//            std::cout << "Plane-&-parallax time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - parallax_time).count() << '\n';
//            std::cout << "RETURN F FROM PL-&-PAR (" << F_best_score.score << ", " << F_best_score.inlier_number << ") -> (" << F_pl_par_score.score << ", " << F_pl_par_score.inlier_number << ")\n";
//#endif
            return true;
        }
//#ifdef DEBUG_DEGENSAC
//        std::cout << "Plane-&-parallax time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - parallax_time).count()
//            << ", score (" << F_pl_par_score.score << ", " << F_pl_par_score.inlier_number << '\n';
//#endif
        non_degenerate_model_score = Score();
#ifdef DEBUG_DEGENSAC
        std::cout << "FAILED TO RECOVER, best #inls " << F_best_score.inlier_number << " calib " << F_calib_score.inlier_number << " " << " " << F_pl_par_score.inlier_number << "\n";
#endif
        return true;
    }

    // RANSAC with plane-and-parallax to find new Fundamental matrix
    bool getFfromTrueK (const Matx33d &H, Mat &F_from_K, Score &F_from_K_score) {
        std::vector<Matx33d> R; std::vector<Vec3d> t;
        if (Utils::decomposeHomography(true_K2_inv * H * true_K1, R, t) == 1) {
            // std::cout << "Warning: translation is zero!\n";
            return false; // is degenerate
        }
        // sign of translation does not make difference
        const Mat F1 = Mat(true_K2_inv_t * Math::getSkewSymmetric(t[0]) * R[0] * true_K1_inv);
        const Mat F2 = Mat(true_K2_inv_t * Math::getSkewSymmetric(t[1]) * R[1] * true_K1_inv);
        const auto score_f1 = quality->getScore(F1), score_f2 = quality->getScore(F2);
        if (score_f1.isBetter(score_f2)) {
            F_from_K = F1; F_from_K_score = score_f1;
        } else {
            F_from_K = F2; F_from_K_score = score_f2;
        }
        return true;
    }
    bool planeAndParallaxRANSAC (const Matx33d &H, std::vector<int> &non_planar_pts, int num_non_planar_pts,
            int max_iters_pl_par, bool use_preemptive, Mat &F_new, Score &F_new_score) {
        if (num_non_planar_pts < 2)
            return false;
        num_models_used_so_far = 0; // reset estimation of lambda for plane-and-parallax
        int max_iters = max_iters_pl_par, non_planar_support = 0, iters = 0;
        std::vector<Matx33d> F_good;
        std::vector<std::pair<int,int>> pairs;
        for (; iters < max_iters; iters++) {
            // draw two random points
            int h_outlier1 = 4 * non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            int h_outlier2 = 4 * non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            while (h_outlier1 == h_outlier2)
                h_outlier2 = 4 * non_planar_pts[rng.uniform(0, num_non_planar_pts)];

            // do plane and parallax with outliers of H
            // F = [(p1' x Hp1) x (p2' x Hp2)]_x H
            const Matx33d F = Math::getSkewSymmetric(
                    (Vec3d(points[h_outlier1+2], points[h_outlier1+3], 1).cross   // p1'
               (H * Vec3d(points[h_outlier1  ], points[h_outlier1+1], 1))).cross // Hp1
                     (Vec3d(points[h_outlier2+2], points[h_outlier2+3], 1).cross   // p2'
               (H * Vec3d(points[h_outlier2  ], points[h_outlier2+1], 1)))       // Hp2
            ) * H;

            const int num_f_inliers_of_h_outliers = getNonPlanarSupport(Mat(F), num_models_used_so_far >= MAX_MODELS_TO_TEST, non_planar_support);
            // std::cout << "pl & par, non planar support " << num_f_inliers_of_h_outliers << " score " << quality->getScore(Mat(F)).score << "\n";
            if (non_planar_support < num_f_inliers_of_h_outliers) {
                non_planar_support = num_f_inliers_of_h_outliers;
                const double predicted_iters = log_conf / log(1 - std::pow
                        (static_cast<double>(num_f_inliers_of_h_outliers) / num_h_outliers, 2));
                if (use_preemptive && ! std::isinf(predicted_iters) && predicted_iters < max_iters)
                    max_iters = static_cast<int>(predicted_iters);
                F_good = { F };
                pairs = {std::make_pair(h_outlier1, h_outlier2)};
            } else if (non_planar_support == num_f_inliers_of_h_outliers) {
                F_good.emplace_back(F);
                pairs.emplace_back(std::make_pair(h_outlier1, h_outlier2));
            }
        }

        if (isFDegenerate(non_planar_support) || F_good.empty()) {
            return false;
        }

        F_new_score = Score();
        int idx = 0, best_idx = 0;
        for (const auto &F_ : F_good) {
            const auto sc = quality->getScore(Mat(F_));
            if (sc.isBetter(F_new_score)) {
                F_new_score = sc;
                F_new = Mat(F_);
            }
            idx++;
        }

#ifdef DEBUG_DEGENSAC
        std::cout << "plane-and-parallax " << H_out_score.score << " " << H_out_score.inlier_number << " out " << non_planar_support_out << " iters " << iters << "\n";
#endif
        return true;
    }
    bool calibDegensac (const Matx33d &H, Mat &F_new, Score &F_new_score, const Mat &F_degen_best, const Score &F_degen_score) {
        if (! is_principal_pt_set) {
            // estimate principal points from coordinates
            float px1 = 0, py1 = 0, px2 = 0, py2 = 0;
            for (int i = 0; i < points_size; i++) {
                const int idx = 4*i;
                if (px1 < points[idx  ]) px1 = points[idx  ];
                if (py1 < points[idx+1]) py1 = points[idx+1];
                if (px2 < points[idx+2]) px2 = points[idx+2];
                if (py2 < points[idx+3]) py2 = points[idx+3];
            }
            setPrincipalPoint((int)(px1/2)+1, (int)(py1/2)+1, (int)(px2/2)+1, (int)(py2/2)+1);
        }
        std::vector<Matx33d> R; std::vector<Vec3d> t; std::vector<Mat> F_good;
        std::vector<double> best_f;
        int non_planar_support_out = 0;
        for (double f = 300; f <= 3500; f += 150.) {
            K(0,0) = K(1,1) = K2(0,0) = K2(1,1) = f;
            const double one_over_f = 1/f;
            K_inv(0,0) = K_inv(1,1) = K2_inv(0,0) = K2_inv(1,1) = K2_inv_t(0,0) = K2_inv_t(1,1) = one_over_f;
            K_inv(0,2) = -K(0,2)*one_over_f; K_inv(1,2) = -K(1,2)*one_over_f;
            K2_inv_t(2,0) = K2_inv(0,2) = -K2(0,2)*one_over_f; K2_inv_t(2,1) = K2_inv(1,2) = -K2(1,2)*one_over_f;
            if (Utils::decomposeHomography(K2_inv * H * K, R, t) == 1) continue;
            Mat F1 = Mat(K2_inv_t * Math::getSkewSymmetric(t[0]) * R[0] * K_inv);
            Mat F2 = Mat(K2_inv_t * Math::getSkewSymmetric(t[1]) * R[1] * K_inv);
            int non_planar_f1 = getNonPlanarSupport(F1, true, non_planar_support_out),
                non_planar_f2 = getNonPlanarSupport(F2, true, non_planar_support_out);
//            std::cout << "f " << f << " non planar supports " << non_planar_f1 << " " << non_planar_f2 << " best " << non_planar_support_out <<
//                      " inliers " << quality->getScore(F1).inlier_number << " "<< quality->getScore(F2).inlier_number << "\n";
            if (non_planar_f1 < non_planar_f2) {
                non_planar_f1 = non_planar_f2; F1 = F2;
            }
            if (non_planar_support_out < non_planar_f1) {
                non_planar_support_out = non_planar_f1;
                F_good = {F1};
                best_f = { f };
            } else if (non_planar_support_out == non_planar_f1) {
                F_good.emplace_back(F1);
                best_f.emplace_back(f);
            }
        }
        F_new_score = Score();
        for (int i = 0; i < (int) F_good.size(); i++) {
            const auto sc = quality->getScore(F_good[i]);
            if (sc.isBetter(F_new_score)) {
                F_new_score = sc;
                F_good[i].copyTo(F_new);
                if (sc.isBetter(best_focal_score)) {
                    best_focal = best_f[i]; // save best focal length
                    best_focal_score = sc;
                }
            }
        }
        if (!F_new_score.isBetter(F_degen_score) && non_planar_support_out <= getNonPlanarSupport(F_degen_best)) {
            return false;
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

#ifdef DEBUG_DEGENSAC
    std::cout << "calib degensac " << F_new_score.score << " " << F_new_score.inlier_number << " out " << non_planar_support_out << "\n";
#endif
        return true;
    }
    void getOutliersH (const Mat &degen_F) {
        // get H outliers
        num_h_outliers = 0;
        plane_parallax_H_sample_size = 0;
        const auto &h_errors = h_reproj_error->getErrors(H_best);
        for (int pt = 0; pt < points_size; pt++)
            if (h_errors[pt] > homography_threshold) {
                h_outliers[num_h_outliers++] = pt;
                if (h_errors[pt] < MAX_H_THR)
                    plane_parallax_H_sample[plane_parallax_H_sample_size++] = pt;
            }
#ifdef DEBUG_DEGENSAC
        std::cout << "H SCORE (" << H_best_score.score << ", " << H_best_score.inlier_number << " H outliers " << num_h_outliers << " / " << points_size << ")\n";
#endif
    }

    bool verifyFundamental (const Mat &F_best, const Score &F_score, const std::vector<bool> &inliers_mask, Mat &F_new, Score &new_score) override {
        if (true_K_given && areSameSingularValues(F_best))
            return false;
        const int f_sample_size = 3, max_H_iters = 5; // 3.52 = log(0.01) / log(1 - std::pow(0.9, 3));
        int num_f_inliers = 0;
        std::vector<int> inliers(points_size), f_sample(f_sample_size);
        for (int i = 0; i < points_size; i++) if (inliers_mask[i]) inliers[num_f_inliers++] = i;
        const auto sampler = UniformSampler::create(0, f_sample_size, num_f_inliers);
        // find e', null space of F^T
        const Vec3d e_prime = Utils::getLeftEpipole(F_best);
        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);
        H_best_score = Score(); H_best = Mat();
        for (int iter = 0; iter < max_H_iters; iter++) {
            sampler->generateSample(f_sample);
            Matx33d H;
            if (!getH(A, e_prime, 4*inliers[f_sample[0]], 4*inliers[f_sample[1]], 4*inliers[f_sample[2]], H))
                continue;
            const auto h_score = h_repr_quality->getScore(Mat(H));
            if (h_score.isBetter(H_best_score)) {
                H_best_score = h_score; H_best = Mat(H);
            }
        }
        if (H_best.empty()) return false; // non-degenerate
        getOutliersH(F_best);
        const bool is_F_degen = true_K_given ? true : isFDegenerate(getNonPlanarSupport(F_best));

#ifdef DEBUG_DEGENSAC
        std::cout << "F final verification: num h outliers " << num_h_outliers << " F non planar support " << F_support << " pts size " << points_size << "\n";
#endif
        Mat F_from_K; Score F_from_K_score;
        bool success = false;
        // generate non-degenerate F even though so-far-the-best one may not be degenerate
        if (true_K_given) {
            // use GT calibration
            if (getFfromTrueK(H_best, F_from_K, F_from_K_score)) {
                new_score = F_from_K_score;
                F_from_K.copyTo(F_new);
                success = true;
            }
        } else {
            // use calibrated DEGENSAC
            if (calibDegensac(H_best, F_from_K, F_from_K_score, F_best, F_score)) {
                new_score = F_from_K_score;
                F_from_K.copyTo(F_new);
                success = true;
            }
        }
        
        if (!is_F_degen) {
#ifdef DEBUG_DEGENSAC
        std::cout << "final F is not degenerate\n";
#endif
            return false;
        } else if (success) // F is degenerate
            return true; // but successfully recovered
        
        // recover degenerate F using plane-and-parallax
        Score plane_parallax_score; Mat F_plane_parallax;
#ifdef DEBUG_DEGENSAC
        std::cout << "verify F: plane-&-parallax score " << plane_parallax_score.score << " " << plane_parallax_score.inlier_number << '\n';
#endif
        if (planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, 20, true, F_plane_parallax, plane_parallax_score)) {
            new_score = plane_parallax_score;
            F_plane_parallax.copyTo(F_new);
            return true;
        }
        // plane-and-parallax failed. A previous non-degenerate so-far-the-best model will be used instead
        new_score = Score();
        return true;
    }
    bool getApproximatedIntrinsics (Mat &_K1, Mat &_K2) override {
        if (best_focal > 0) {
            K(0,0) = K(1,1) = K2(0,0) = K2(1,1) = best_focal;
            _K1 = Mat(K); _K2 = Mat(K2);
            return true;
        }
        return false;
    }
    void setPrincipalPoint (double px_, double py_) override {
        setPrincipalPoint(px_, py_, 0, 0);
    }
    void setPrincipalPoint (double px_, double py_, double px2_, double py2_) override {
        if (px_ > DBL_EPSILON && py_ > DBL_EPSILON) {
            is_principal_pt_set = true;
            K = {1, 0, px_, 0, 1, py_, 0, 0, 1};
            if (px2_ > DBL_EPSILON && py2_ > DBL_EPSILON) K2 = {1, 0, px2_, 0, 1, py2_, 0, 0, 1};
            else K2 = K;
            K_inv = K2_inv = K2_inv_t = Matx33d::eye();
        }
    }
private:
    static bool areSameSingularValuesEssential (const Matx33d &E) {
        const auto EtE = E.t() * E;
        const auto e11 = EtE(0,0), e12 = EtE(0,1), e13 = EtE(0,2), e22 = EtE(1,1), e23 = EtE(1,2), e33 = EtE(2,2);
        double // c1 = -e33*e12*e12 + 2*e12*e13*e23 - e22*e13*e13 - e11*e23*e23 + e11*e22*e33 == 0 because F has 0 third singular value
        c2 = e12*e12 + e13*e13 + e23*e23 - e11*e22 - e11*e33 - e22*e33, c3 = e11 + e22 + e33;// c4 = -1;
        double d = c3 * c3 + 4 * c2; d = d < 0 ? 0 : sqrt(d); // numerical problem, eigen values can't be complex for positive definie E^T E matrix
        // check eigen ratio. eig value is squared singular value. Use 5% threshold: 1.05^2 = 1.1025
        // Vec3d w; SVD::compute(E, w); std::cout << "SVD: " << w / w[1] << " eig " << sqrt((-c3 - d) / (-c3 + d)) << "\n"; // debug
        return (-c3 - d) / (-c3 + d) < 1.1025;
    }
    bool getH (const Matx33d &A, const Vec3d &e_prime, int smpl1, int smpl2, int smpl3, Matx33d &H) {
        Vec3d p1(points[smpl1  ], points[smpl1+1], 1), p2(points[smpl2  ], points[smpl2+1], 1), p3(points[smpl3  ], points[smpl3+1], 1);
        Vec3d P1(points[smpl1+2], points[smpl1+3], 1), P2(points[smpl2+2], points[smpl2+3], 1), P3(points[smpl3+2], points[smpl3+3], 1);
        const Matx33d M (p1[0], p1[1], 1, p2[0], p2[1], 1, p3[0], p3[1], 1);
        if (p1.cross(p2).dot(p3) * P1.cross(P2).dot(P3) < 0) return false;
        // (x′i × e')
        const Vec3d P1e = P1.cross(e_prime), P2e = P2.cross(e_prime), P3e = P3.cross(e_prime);
        // x′i × (A xi))^T (x′i × e′) / ‖x′i×e′‖^2,
        const Vec3d b (P1.cross(A * p1).dot(P1e) / (P1e[0]*P1e[0]+P1e[1]*P1e[1]+P1e[2]*P1e[2]),
                       P2.cross(A * p2).dot(P2e) / (P2e[0]*P2e[0]+P2e[1]*P2e[1]+P2e[2]*P2e[2]),
                       P3.cross(A * p3).dot(P3e) / (P3e[0]*P3e[0]+P3e[1]*P3e[1]+P3e[2]*P3e[2]));

        H = A - e_prime * (M.inv() * b).t();
        return true;
    }
    int getNonPlanarSupport (const Mat &F, bool preemptive=false, int max_so_far=0) {
        int non_rand_support = 0;
        f_error->setModelParameters(F);
        if (preemptive) {
            const auto preemptive_thr = -num_h_outliers + max_so_far;
            for (int pt = 0; pt < num_h_outliers; pt++)
                if (f_error->getError(h_outliers[pt]) < f_threshold_sqr)
                    non_rand_support++;
                else if (non_rand_support - pt < preemptive_thr)
                        break;
        } else {
            for (int pt = 0; pt < num_h_outliers; pt++)
                if (f_error->getError(h_outliers[pt]) < f_threshold_sqr)
                    non_rand_support++;
            if (num_models_used_so_far < MAX_MODELS_TO_TEST && !true_K_given/*for K we know that recovered F cannot be degenerate*/) {
                non_planar_supports[num_models_used_so_far++] = non_rand_support;
                if (num_models_used_so_far == MAX_MODELS_TO_TEST) {
                    getLambda(non_planar_supports, 2.32, num_h_outliers, 0, false, estimated_min_non_planar_support);
                    if (estimated_min_non_planar_support < 3) estimated_min_non_planar_support = 3;
#ifdef DEBUG_DEGENSAC
                    std::cout << "est support " << estimated_min_non_planar_support << "\n";
#endif
                }
            }
        }
        return non_rand_support;
    }
    inline bool isModelValid(const Mat &F, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(F, sample);
    }
    bool areSameSingularValues (const Matx33d &F) {
        return areSameSingularValuesEssential(true_K2_t * F * true_K1);
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
