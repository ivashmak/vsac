#include "precomp.hpp"

namespace cv { namespace vsac {
class CovarianceHomographySolverImpl : public CovarianceHomographySolver {
private:
    Matx33d T1, T2;
    Mat norm_pts;
    float * norm_points;
    std::vector<bool> mask;
    int points_size;
    double covariance[81] = {0};
    double * t1, * t2;
public:
    explicit CovarianceHomographySolverImpl (const Mat &points_) {
        points_size = points_.rows;
        std::vector<int> sample(points_size);
        for (int i = 0; i < points_size; i++) sample[i] = i;
        const Ptr<NormTransform> normTr = NormTransform::create(points_);
        normTr->getNormTransformation(norm_pts, sample, points_size, T1, T2);
        norm_points = (float *) norm_pts.data;
        t1 = T1.val;
        t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }
    void reset () override {
        std::fill(covariance, covariance+81, 0);
        std::fill(mask.begin(), mask.end(), false);
    }

    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &/*weights*/) override {
        double a1[9] = {0, 0, -1, 0, 0, 0, 0, 0, 0},
               a2[9] = {0, 0, 0, 0, 0, -1, 0, 0, 0};

        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                a1[0] = -x1;
                a1[1] = -y1;
                a1[6] = x2*x1;
                a1[7] = x2*y1;
                a1[8] = x2;

                a2[3] = -x1;
                a2[4] = -y1;
                a2[6] = y2*x1;
                a2[7] = y2*y1;
                a2[8] = y2;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] +=-a1[j]*a1[z] - a2[j]*a2[z];
                else
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                covariance[j*9+z] = covariance[z*9+j];

#ifdef HAVE_EIGEN
        Mat H = Mat_<double>(3,3);
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Eigen::Matrix<double, 9, 9>
                (Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> (
                        (Eigen::Matrix<double, 9, 9> (covariance))).householderQ()).col(8);
#else
       Matx<double, 9, 9> Vt;
       Vec<double, 9> D;
       if (! eigen(Matx<double, 9, 9>(covariance), D, Vt)) return 0;
       Mat H = Mat_<double>(3, 3, Vt.val + 72/*=8*9*/);
#endif

        const auto * const h = (double *) H.data;
        // H = T2^-1 H T1
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(h[0]/t2[0] - (h[6]*t2[2])/t2[0]),
           t1[0]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]), h[2]/t2[0] + t1[2]*(h[0]/t2[0] -
           (h[6]*t2[2])/t2[0]) + t1[5]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]) - (h[8]*t2[2])/t2[0],
           t1[0]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]), t1[0]*(h[4]/t2[0] - (h[7]*t2[5])/t2[0]),
           h[5]/t2[0] + t1[2]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]) + t1[5]*(h[4]/t2[0] -
           (h[7]*t2[5])/t2[0]) - (h[8]*t2[5])/t2[0], t1[0]*h[6], t1[0]*h[7],
           h[8] + h[6]*t1[2] + h[7]*t1[5])) };

        return 1;
    }
};
Ptr<CovarianceHomographySolver> CovarianceHomographySolver::create (const Mat &points) {
    return makePtr<CovarianceHomographySolverImpl>(points);
}


/*class CovarianceAffineSolverImpl : public CovarianceAffineSolver {
private:
    Matx33d T1, T2;
    Mat norm_pts;
    float * norm_points;
    std::vector<bool> mask;
    int points_size;
    double covariance[36] = {0};
    double * t1, * t2;
public:
    explicit CovarianceAffineSolverImpl (const Mat &points_) {
        points_size = points_.rows;
        std::vector<int> sample(points_size);
        for (int i = 0; i < points_size; i++) sample[i] = i;
        const Ptr<NormTransform> normTr = NormTransform::create(points_);
        normTr->getNormTransformation(norm_pts, sample, points_size, T1, T2);
        norm_points = (float *) norm_pts.data;
        t1 = T1.val;
        t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }
    void reset () override {
        std::fill(covariance, covariance+36, 0);
        std::fill(mask.begin(), mask.end(), false);
    }

    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &) override {
        double r1[6] = {0, 0, 1, 0, 0, 0}; // row 1 of A
        double r2[6] = {0, 0, 0, 0, 0, 1}; // row 2 of A

        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                r1[0] = x1;
                r1[1] = y1;

                r2[3] = x1;
                r2[4] = y1;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] +=-a1[j]*a1[z] - a2[j]*a2[z];
                else
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                covariance[j*9+z] = covariance[z*9+j];

#ifdef HAVE_EIGEN
        Mat H = Mat_<double>(3,3);
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Eigen::Matrix<double, 9, 9>
                (Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> (
                        (Eigen::Matrix<double, 9, 9> (covariance))).householderQ()).col(8);
#else
       Matx<double, 9, 9> Vt;
       Vec<double, 9> D;
       if (! eigen(Matx<double, 9, 9>(covariance), D, Vt)) return 0;
       Mat H = Mat_<double>(3, 3, Vt.val + 72);
#endif

        const auto * const h = (double *) H.data;
        // H = T2^-1 H T1
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(h[0]/t2[0] - (h[6]*t2[2])/t2[0]),
           t1[0]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]), h[2]/t2[0] + t1[2]*(h[0]/t2[0] -
           (h[6]*t2[2])/t2[0]) + t1[5]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]) - (h[8]*t2[2])/t2[0],
           t1[0]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]), t1[0]*(h[4]/t2[0] - (h[7]*t2[5])/t2[0]),
           h[5]/t2[0] + t1[2]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]) + t1[5]*(h[4]/t2[0] -
           (h[7]*t2[5])/t2[0]) - (h[8]*t2[5])/t2[0], t1[0]*h[6], t1[0]*h[7],
           h[8] + h[6]*t1[2] + h[7]*t1[5])) };

        return 1;
    }
};
Ptr<CovarianceAffineSolver> CovarianceAffineSolver::create (const Mat &points) {
    return makePtr<CovarianceAffineSolverImpl>(points);
}*/

class CovarianceFundamentalSolverImpl : public CovarianceFundamentalSolver {
private:
    Matx33d T1, T2;
    Mat norm_pts;
    float * norm_points;
    std::vector<bool> mask;
    int points_size;
    double covariance[81] = {0};
    double * t1, * t2;
public:
    explicit CovarianceFundamentalSolverImpl (const Mat &points_) {
        points_size = points_.rows;
        std::vector<int> sample(points_size);
        for (int i = 0; i < points_size; i++) sample[i] = i;
        const Ptr<NormTransform> normTr = NormTransform::create(points_);
        normTr->getNormTransformation(norm_pts, sample, points_size, T1, T2);
        norm_points = (float *) norm_pts.data;
        t1 = T1.val;
        t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }

    void reset () override {
        std::fill(covariance, covariance+81, 0);
        std::fill(mask.begin(), mask.end(), false);
    }
    /*
     * Find Homography matrix using (weighted) non-minimal estimation.
     * Use Principal Component Analysis. Use normalized points.
     */
    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &/*weights*/) override {
        double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};

        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                a[0] = x2*x1;
                a[1] = x2*y1;
                a[2] = x2;
                a[3] = y2*x1;
                a[4] = y2*y1;
                a[5] = y2;
                a[6] = x1;
                a[7] = y1;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] -= a[j]*a[z];
                else
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] += a[j]*a[z];
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                covariance[j*9+z] = covariance[z*9+j];

#ifdef HAVE_EIGEN
        models = std::vector<Mat>{ Mat_<double>(3,3) };
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)models[0].data) = Eigen::JacobiSVD
                <Eigen::Matrix<double, 9, 9>> ((Eigen::Matrix<double, 9, 9>(covariance)),
                        Eigen::ComputeFullV).matrixV().col(8);
#else
       Matx<double, 9, 9> AtA_(covariance), U, Vt;
       Vec<double, 9> W;
       SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
       models = std::vector<Mat> { Mat_<double>(3, 3, Vt.val + 72 /*=8*9*/) };
#endif
        FundamentalDegeneracy::recoverRank(models[0], true/*F*/);

        const auto * const f = (double *) models[0].data;
        // F = T2^T F T1
        models[0] = Mat(Matx33d(t1[0]*t2[0]*f[0],t1[0]*t2[0]*f[1], t2[0]*f[2] + t2[0]*f[0]*t1[2] +
            t2[0]*f[1]*t1[5], t1[0]*t2[0]*f[3],t1[0]*t2[0]*f[4], t2[0]*f[5] + t2[0]*f[3]*t1[2] +
            t2[0]*f[4]*t1[5], t1[0]*(f[6] + f[0]*t2[2] + f[3]*t2[5]), t1[0]*(f[7] + f[1]*t2[2] +
            f[4]*t2[5]), f[8] + t1[2]*(f[6] + f[0]*t2[2] + f[3]*t2[5]) + t1[5]*(f[7] + f[1]*t2[2] +
            f[4]*t2[5]) + f[2]*t2[2] + f[5]*t2[5]));
        return 1;
    }
};
Ptr<CovarianceFundamentalSolver> CovarianceFundamentalSolver::create (const Mat &points) {
    return makePtr<CovarianceFundamentalSolverImpl>(points);
}

class CovarianceEssentialSolverImpl : public CovarianceEssentialSolver {
private:
    float * points;
    std::vector<bool> mask;
    int points_size;
    double covariance[81] = {0};
    bool enforce = true;
public:
    void setEnforceRankConstraint (bool enforce_) override { enforce = enforce_; }
    explicit CovarianceEssentialSolverImpl (const Mat &points_) {
        points_size = points_.rows;
        std::vector<int> sample(points_size);
        for (int i = 0; i < points_size; i++) sample[i] = i;
        points = (float *) points_.data;
        mask = std::vector<bool>(points_size, false);
    }

    void reset () override {
        std::fill(covariance, covariance+81, 0);
        std::fill(mask.begin(), mask.end(), false);
    }
    /*
     * Find Homography matrix using (weighted) non-minimal estimation.
     * Use Principal Component Analysis. Use normalized points.
     */
    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &/*weights*/) override {
        double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};

        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = points[smpl  ], y1 = points[smpl+1],
                        x2 = points[smpl+2], y2 = points[smpl+3];

                a[0] = x2*x1;
                a[1] = x2*y1;
                a[2] = x2;
                a[3] = y2*x1;
                a[4] = y2*y1;
                a[5] = y2;
                a[6] = x1;
                a[7] = y1;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] -= a[j]*a[z];
                else
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] += a[j]*a[z];
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                covariance[j*9+z] = covariance[z*9+j];

#ifdef HAVE_EIGEN
        models = std::vector<Mat>{ Mat_<double>(3,3) };
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)models[0].data) = Eigen::JacobiSVD
                <Eigen::Matrix<double, 9, 9>> ((Eigen::Matrix<double, 9, 9>(covariance)),
                                               Eigen::ComputeFullV).matrixV().col(8);
#else
        Matx<double, 9, 9> AtA_(covariance), U, Vt;
        Vec<double, 9> W;
        SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
        models = std::vector<Mat> { Mat_<double>(3, 3, Vt.val + 72 /*=8*9*/) };
#endif
        if (enforce)
            FundamentalDegeneracy::recoverRank(models[0], false/*E*/);

        const auto * const f = (double *) models[0].data;
        return 1;
    }
};
Ptr<CovarianceEssentialSolver> CovarianceEssentialSolver::create (const Mat &points) {
    return makePtr<CovarianceEssentialSolverImpl>(points);
}


class CovariancePolisherImpl : public CovariancePolisher {
    int lsq_iters;
    const Ptr<Degeneracy> degeneracy;
    const Ptr<Quality> quality;
    const Ptr<CovarianceSolver> solver;
    const Ptr<Error> error_fnc;
    std::vector<bool> inliers_mask, new_inliers_mask;
    std::vector<Mat> models;
    std::vector<double> weights;
    double threshold;
    bool filter_pts;
public:
    CovariancePolisherImpl (const Ptr<Degeneracy> &degeneracy_, const Ptr<Quality> &quality_, const Ptr<CovarianceSolver> &solver_, int lsq_iters_, bool filter_pts_) :
        degeneracy(degeneracy_), quality(quality_), solver(solver_), error_fnc(quality_->getErrorFnc()) {
        lsq_iters = lsq_iters_;
        threshold = quality->getThreshold();
        inliers_mask = std::vector<bool>(quality->getPointsSize());
        new_inliers_mask = std::vector<bool>(quality->getPointsSize());
        filter_pts = filter_pts_;
    }
    void setInlierThreshold (double thr) override { threshold = thr*thr; } 
    bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
                                  Mat &new_model, Score &new_model_score) override {
        // quality->getInliers(model, inliers_mask);
        Quality::getInliers(error_fnc, model, inliers_mask, threshold);
        if (filter_pts)
             degeneracy->filterInliers(model, inliers_mask);
        new_model_score = best_model_score;
        Score best_score_polisher;
        Mat best_model_polisher;
        for (int iter = 0; iter < lsq_iters; iter++) {
            bool is_updated = false;
            const int num_sols = solver->estimate(inliers_mask, models, weights);
            for (int i = 0; i < num_sols; i++) {
                const auto &errors = error_fnc->getErrors(models[i]);
                const auto score = quality->getScore(errors);
                // std::cout << "iter " << iter << " cov score " << score.score << " #inls " << score.inlier_number << "\n";
                if (score.isBetter(new_model_score)) {
                    new_model_score = score;
                    models[i].copyTo(new_model);
                    Quality::getInliers(errors, new_inliers_mask, threshold);
                    if (filter_pts)
                         degeneracy->filterInliers(model, inliers_mask);
                    if (Utils::intersectionOverUnion(new_inliers_mask, inliers_mask) >= 0.99)
                        break;
                    inliers_mask = new_inliers_mask;
                    is_updated = true;
                }
                if (score.isBetter(best_score_polisher)) {
                    best_score_polisher = score;
                    models[i].copyTo(best_model_polisher);
                }
            }
            if (!is_updated)
                break;
        }
        if (new_model.empty()) {
            new_model_score = best_score_polisher;
            best_model_polisher.copyTo(new_model);
        }
        return true;
//        return !new_model.empty();
    }
};
Ptr<CovariancePolisher> CovariancePolisher::create (const Ptr<Degeneracy> &degeneracy, const Ptr<Quality> &quality_, const Ptr<CovarianceSolver> &solver_, int lsq_iters, bool filter_pts_) {
    return makePtr<CovariancePolisherImpl>(degeneracy, quality_, solver_, lsq_iters, filter_pts_);
}

class WeightedPolisherImpl : public WeightedPolisher {
    const Ptr<Degeneracy> degeneracy;
    const Ptr<Quality> quality;
    const Ptr<NonMinimalSolver> non_min_solver;
    const Ptr<Error> error_fnc;
    const Ptr<GammaValues> gamma_generator;
    std::vector<Mat> models;
    std::vector<int> inliers;
    std::vector<double> weights;
    const std::vector<double> &stored_gamma_values;
    double threshold, max_sigma_sqr, scale_of_stored_gammas, one_over_sigma, gamma_k, squared_sigma_max_2;
    int points_size, number_of_irwls_iters, stored_gamma_number_min1;
public:
    WeightedPolisherImpl (const Ptr<Degeneracy> &degeneracy_, const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &solver_, const Ptr<Error> &error_fnc_,
          const Ptr<GammaValues> &gamma_generator_, int number_of_irwls_iters_, int DoF, double upper_incomplete_of_sigma_quantile, double C, double max_sigma) :
            degeneracy (degeneracy_), quality(quality_), non_min_solver(solver_), error_fnc(error_fnc_), gamma_generator(gamma_generator_),
            stored_gamma_values (gamma_generator_->getGammaValues()) {
        threshold = quality->getThreshold();
        points_size = quality_->getPointsSize();

        inliers = std::vector<int>(points_size);
        weights = std::vector<double>(points_size);

        gamma_k = upper_incomplete_of_sigma_quantile;
        number_of_irwls_iters = number_of_irwls_iters_;
        double dof_minus_one_per_two = (DoF - 1.0) / 2.0;
        double two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
        double C_times_two_ad_dof = C * two_ad_dof;
        // Calculate 2 * \sigma_{max}^2 a priori
        squared_sigma_max_2 = max_sigma * max_sigma * 2.0;
        // Divide C * 2^(DoF - 1) by \sigma_{max} a priori
        one_over_sigma = C_times_two_ad_dof / max_sigma;
        max_sigma_sqr = squared_sigma_max_2 * 0.5;
        stored_gamma_number_min1 = gamma_generator->getTableSize()-1;
        scale_of_stored_gammas = gamma_generator->getScaleOfGammaValues();
    }

    bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
                                  Mat &new_model, Score &new_model_score) override {
        Mat polished_model;
        model.copyTo(polished_model);
        // Do the iteratively re-weighted least squares fitting
        for (int iterations = 0; iterations < number_of_irwls_iters; iterations++) {
            // error->setModelParameters(polished_model);
            error_fnc->setModelParameters(polished_model);
            // Remove everything from the residual vector

            int sigma_inliers_cnt = 0;

            // Collect the points which are closer than the maximum threshold
            for (int point_idx = 0; point_idx < points_size; ++point_idx) {
                // Calculate the residual of the current point
                const double sqr_residual = error_fnc->getError(point_idx);
                if (sqr_residual < max_sigma_sqr) {
                    // Get the position of the gamma value in the lookup table
                    int x = (int) round(scale_of_stored_gammas * sqr_residual / squared_sigma_max_2);

                    // If the sought gamma value is not stored in the lookup, return the closest element
                    if (x >= stored_gamma_number_min1 || x < 0 /*overflow*/) // actual number of gamma values is 1 more, so >=
                        x = stored_gamma_number_min1;

                    inliers[sigma_inliers_cnt] = point_idx; // store index of point for LSQ
                    weights[sigma_inliers_cnt++] = one_over_sigma * (stored_gamma_values[x] - gamma_k);
                }
            }
            // std::vector<int> new_inliers = inliers;
            // const int num_new_inliers = degeneracy->filterInliers(polished_model, new_inliers, sigma_inliers_cnt);
            // for (int p = 0; p < num_new_inliers; p++) {
            //     if (new_inliers[p] != inliers[p]) {
            //         std::swap(weights[p], weights[--sigma_inliers_cnt]);
            //         // test
            //         std::swap(inliers[p], inliers[sigma_inliers_cnt]);
            //         assert(new_inliers[p] == inliers[p]);
            //     }
            // }
            // inliers = new_inliers;
            // sigma_inliers_cnt = num_new_inliers;

            const int num_est_models = non_min_solver->estimate(inliers, sigma_inliers_cnt, models, weights);
            if (num_est_models == 0)
                break; // break iterations

            Mat best_wlsq_model;
            Score best_wlsq_model_score;
            for (int m = 0; m < num_est_models; m++) {
                const Score sc = quality->getScore(models[m]);
                if (sc.isBetter(best_wlsq_model_score)) {
                    best_wlsq_model = models[m].clone();
                    best_wlsq_model_score = sc;
                }
            }

            polished_model = best_wlsq_model;

            if (best_wlsq_model_score.isBetter(new_model_score)){
                new_model_score = best_wlsq_model_score;
                polished_model.copyTo(new_model);
            }
        }
        return true;
    }
};
Ptr<WeightedPolisher> WeightedPolisher::create (const Ptr<Degeneracy> &degeneracy, const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &solver_, const Ptr<Error> &error_fnc_,
            const Ptr<GammaValues> &gamma_generator_, int number_of_irwls_iters_, int DoF, double upper_incomplete_of_sigma_quantile, double C, double max_sigma) {
    return makePtr<WeightedPolisherImpl>(degeneracy, quality_, solver_, error_fnc_, gamma_generator_, number_of_irwls_iters_, DoF, upper_incomplete_of_sigma_quantile, C, max_sigma);
}

}}