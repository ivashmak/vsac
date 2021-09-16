#include "precomp.hpp"
#include "polynom_solver.h"

namespace cv { namespace vsac {
class FundamentalMinimalSolver7ptsImpl: public FundamentalMinimalSolver7pts {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit FundamentalMinimalSolver7ptsImpl (const Mat &points_) :
            points_mat (&points_), points ((float *) points_.data) {}

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int m = 7, n = 9; // rows, cols
        std::vector<double> a(63); // m*n
        auto * a_ = &a[0];

        for (int i = 0; i < m; i++ ) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl  ], y1 = points[smpl+1],
                       x2 = points[smpl+2], y2 = points[smpl+3];

            (*a_++) = x2*x1;
            (*a_++) = x2*y1;
            (*a_++) = x2;
            (*a_++) = y2*x1;
            (*a_++) = y2*y1;
            (*a_++) = y2;
            (*a_++) = x1;
            (*a_++) = y1;
            (*a_++) = 1;
        }

        if (!Math::eliminateUpperTriangular(a, m, n))
            return 0;

        /*
         [a11 a12 a13 a14 a15 a16 a17 a18 a19]
         [  0 a22 a23 a24 a25 a26 a27 a28 a29]
         [  0   0 a33 a34 a35 a36 a37 a38 a39]
         [  0   0   0 a44 a45 a46 a47 a48 a49]
         [  0   0   0   0 a55 a56 a57 a58 a59]
         [  0   0   0   0   0 a66 a67 a68 a69]
         [  0   0   0   0   0   0 a77 a78 a79]
         */
        double f1[9], f2[9];

        f1[8] = 1.;
        f1[7] = 0.;
        f1[6] = -a[6*n+8] / a[6*n+6];

        f2[8] = 0.;
        f2[7] = -a[6*n+6] / a[6*n+7];
        f2[6] = 1;

        // start from the last row
        for (int i = m-2; i >= 0; i--) {
            const int row_i = i*n;
            double acc1 = 0, acc2 = 0;
            for (int j = i+1; j < n; j++) {
                acc1 -= a[row_i + j] * f1[j];
                acc2 -= a[row_i + j] * f2[j];
            }
            f1[i] = acc1 / a[row_i + i];
            f2[i] = acc2 / a[row_i + i];

            if (std::isnan(f1[i]) || std::isnan(f2[i]))
                return 0; // due to numerical errors return 0 solutions
        }
        // OpenCV:
        double c[4], r[3];
        double t0 = 0, t1 = 0, t2 = 0;

        for (int i = 0; i < 9; i++)
            f1[i] -= f2[i];

        t0 = f2[4]*f2[8] - f2[5]*f2[7];
        t1 = f2[3]*f2[8] - f2[5]*f2[6];
        t2 = f2[3]*f2[7] - f2[4]*f2[6];

        c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

        c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
               f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
               f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
               f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
               f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
               f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
               f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

        t0 = f1[4]*f1[8] - f1[5]*f1[7];
        t1 = f1[3]*f1[8] - f1[5]*f1[6];
        t2 = f1[3]*f1[7] - f1[4]*f1[6];

        c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
               f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
               f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
               f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
               f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
               f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
               f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

        c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

        // solve the cubic equation; there can be 1 to 3 roots ...
        const int nroots = solve_deg3(c[0], c[1], c[2], c[3], r[0], r[1], r[2]);
        if (nroots < 1) return 0;

        models = std::vector<Mat>(nroots);
        for (int k = 0; k < nroots; k++) {
            models[k] = Mat_<double>(3,3);
            auto * F_ptr = (double *) models[k].data;

            // for each root form the fundamental matrix
            double lambda = r[k], mu = 1;
            double s = f1[8]*lambda + f2[8];

            // normalize each matrix, so that F(3,3) (~F[8]) == 1
            if (fabs(s) > FLT_EPSILON) {
                mu = 1/s;
                lambda *= mu;
                F_ptr[8] = 1;
            } else
                F_ptr[8] = 0;

            for (int i = 0; i < 8; i++)
                F_ptr[i] = f1[i] * lambda + f2[i] * mu;
        }
        return nroots;
    }

    int getMaxNumberOfSolutions () const override { return 3; }
    int getSampleSize() const override { return 7; }
};
Ptr<FundamentalMinimalSolver7pts> FundamentalMinimalSolver7pts::create(const Mat &points_) {
    return makePtr<FundamentalMinimalSolver7ptsImpl>(points_);
}

class FundamentalSVDSolverImpl: public FundamentalSVDSolver {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit FundamentalSVDSolverImpl (const Mat &points_) :
            points_mat (&points_), points ((float *) points_.data) {}

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int m = 7, n = 9; // rows, cols
        std::vector<double> a(m*n);
        auto * a_ = &a[0];

        for (int i = 0; i < m; i++ ) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl  ], y1 = points[smpl+1],
                    x2 = points[smpl+2], y2 = points[smpl+3];

            (*a_++) = x2*x1;
            (*a_++) = x2*y1;
            (*a_++) = x2;
            (*a_++) = y2*x1;
            (*a_++) = y2*y1;
            (*a_++) = y2;
            (*a_++) = x1;
            (*a_++) = y1;
            (*a_++) = 1;
        }

        Mat U, Vt, D;
        cv::Matx<double, 7, 9> A(&a[0]);
        SVD::compute(A, D, U, Vt, SVD::FULL_UV+SVD::MODIFY_A);

        double * f1 = (double *) Vt.row(8).data, * f2 = (double *) Vt.row(7).data;

        // OpenCV:
        double c[4], r[3];
        double t0, t1, t2;
        Mat_<double> coeffs (1, 4, c);
        Mat_<double> roots (1, 3, r);

        for (int i = 0; i < 9; i++)
            f1[i] -= f2[i];

        t0 = f2[4]*f2[8] - f2[5]*f2[7];
        t1 = f2[3]*f2[8] - f2[5]*f2[6];
        t2 = f2[3]*f2[7] - f2[4]*f2[6];

        c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

        c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
               f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
               f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
               f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
               f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
               f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
               f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

        t0 = f1[4]*f1[8] - f1[5]*f1[7];
        t1 = f1[3]*f1[8] - f1[5]*f1[6];
        t2 = f1[3]*f1[7] - f1[4]*f1[6];

        c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
               f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
               f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
               f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
               f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
               f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
               f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

        c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

        // solve the cubic equation; there can be 1 to 3 roots ...
        int nroots = solveCubic (coeffs, roots);
        if (nroots < 1) return 0;

        models = std::vector<Mat>(nroots);
        for (int k = 0; k < nroots; k++) {
            models[k] = Mat_<double>(3,3);
            auto * F_ptr = (double *) models[k].data;

            // for each root form the fundamental matrix
            double lambda = r[k], mu = 1;
            double s = f1[8]*lambda + f2[8];

            // normalize each matrix, so that F(3,3) (~F[8]) == 1
            if (fabs(s) > FLT_EPSILON) {
                mu = 1/s;
                lambda *= mu;
                F_ptr[8] = 1;
            } else
                F_ptr[8] = 0;

            for (int i = 0; i < 8; i++)
                F_ptr[i] = f1[i] * lambda + f2[i] * mu;
        }
        return nroots;
    }

    int getMaxNumberOfSolutions () const override { return 3; }
    int getSampleSize() const override { return 7; }
};
Ptr<FundamentalSVDSolver> FundamentalSVDSolver::create(const Mat &points_) {
    return makePtr<FundamentalSVDSolverImpl>(points_);
}

class FundamentalMinimalSolver8ptsImpl : public FundamentalMinimalSolver8pts {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit FundamentalMinimalSolver8ptsImpl (const Mat &points_) :
            points_mat (&points_), points ((float*) points_.data)
    { CV_DbgAssert(points); }

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int m = 8, n = 9; // rows, cols
        std::vector<double> a(72); // m*n
        auto * a_ = &a[0];

        for (int i = 0; i < m; i++ ) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl  ], y1 = points[smpl+1],
                       x2 = points[smpl+2], y2 = points[smpl+3];

            (*a_++) = x2*x1;
            (*a_++) = x2*y1;
            (*a_++) = x2;
            (*a_++) = y2*x1;
            (*a_++) = y2*y1;
            (*a_++) = y2;
            (*a_++) = x1;
            (*a_++) = y1;
            (*a_++) = 1;
        }

        if (!Math::eliminateUpperTriangular(a, m, n))
            return 0;

        /*
         [a11 a12 a13 a14 a15 a16 a17 a18 a19]
         [  0 a22 a23 a24 a25 a26 a27 a28 a29]
         [  0   0 a33 a34 a35 a36 a37 a38 a39]
         [  0   0   0 a44 a45 a46 a47 a48 a49]
         [  0   0   0   0 a55 a56 a57 a58 a59]
         [  0   0   0   0   0 a66 a67 a68 a69]
         [  0   0   0   0   0   0 a77 a78 a79]
         [  0   0   0   0   0   0   0 a88 a89]

         f9 = 1
         f8 = (-a89*f9) / a88
         f7 = (-a79*f9 - a78*f8) / a77
         f6 = (-a69*f9 - a68*f8 - a69*f9) / a66
         ...
         */

        models = std::vector<Mat>{ Mat_<double>(3,3) };
        auto * f = (double *) models[0].data;
        f[8] = 1.;

        // start from the last row
        for (int i = m-1; i >= 0; i--) {
            double acc = 0;
            for (int j = i+1; j < n; j++)
                acc -= a[i*n+j]*f[j];

            f[i] = acc / a[i*n+i];
            // due to numerical errors return 0 solutions
            if (std::isnan(f[i]))
                return 0;
        }
        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 8; }
};
Ptr<FundamentalMinimalSolver8pts> FundamentalMinimalSolver8pts::create(const Mat &points_) {
    return makePtr<FundamentalMinimalSolver8ptsImpl>(points_);
}

class EpipolarNonMinimalSolverImpl : public EpipolarNonMinimalSolver {
private:
    const Mat * points_mat;
    Matx33d _T1, _T2;
    Ptr<NormTransform> normTr = nullptr;
    bool enforce_rank = true, is_fundamental;
public:
    explicit EpipolarNonMinimalSolverImpl (const Mat &points_, const Matx33d &T1, const Matx33d &T2)
        : points_mat(&points_), _T1(T1), _T2(T2) {
        is_fundamental = true;
    }
    explicit EpipolarNonMinimalSolverImpl (const Mat &points_, bool is_fundamental_) :
        points_mat(&points_) {
        is_fundamental = is_fundamental_;
        if (is_fundamental)
            normTr = NormTransform::create(points_);
    }
    void enforceRankConstraint (bool enforce) override { enforce_rank = enforce; }
    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat>
            &models, const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Matx33d T1, T2;
        Mat norm_points;
        if (normTr)
            normTr->getNormTransformation(norm_points, sample, sample_size, T1, T2);
        const auto * const norm_pts = normTr ? (float *) norm_points.data : (float *) points_mat->data;

        // ------- 8 points algorithm with Eigen and covariance matrix --------------
        double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1}, AtA[81] = {0}; // 9x9

        if (weights.empty()) {
            for (int i = 0; i < sample_size; i++) {
                const int norm_points_idx = 4*i;
                const double x1 = norm_pts[norm_points_idx  ], y1 = norm_pts[norm_points_idx+1],
                             x2 = norm_pts[norm_points_idx+2], y2 = norm_pts[norm_points_idx+3];
                a[0] = x2*x1;
                a[1] = x2*y1;
                a[2] = x2;
                a[3] = y2*x1;
                a[4] = y2*y1;
                a[5] = y2;
                a[6] = x1;
                a[7] = y1;

                // calculate covariance for eigen
                for (int row = 0; row < 9; row++)
                    for (int col = row; col < 9; col++)
                        AtA[row*9+col] += a[row]*a[col];
            }
        } else {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const double weight = weights[i];
                const double x1 = norm_pts[smpl  ], y1 = norm_pts[smpl+1],
                             x2 = norm_pts[smpl+2], y2 = norm_pts[smpl+3];
                const double weight_times_x2 = weight * x2,
                             weight_times_y2 = weight * y2;

                a[0] = weight_times_x2 * x1;
                a[1] = weight_times_x2 * y1;
                a[2] = weight_times_x2;
                a[3] = weight_times_y2 * x1;
                a[4] = weight_times_y2 * y1;
                a[5] = weight_times_y2;
                a[6] = weight * x1;
                a[7] = weight * y1;
                a[8] = weight;

                // calculate covariance for eigen
                for (int row = 0; row < 9; row++)
                    for (int col = row; col < 9; col++)
                        AtA[row*9+col] += a[row]*a[col];
            }
        }

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                AtA[j*9+z] = AtA[z*9+j];

#ifdef HAVE_EIGEN
        models = std::vector<Mat>{ Mat_<double>(3,3) };
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)models[0].data) = Eigen::JacobiSVD
                <Eigen::Matrix<double, 9, 9>> ((Eigen::Matrix<double, 9, 9>(AtA)),
                        Eigen::ComputeFullV).matrixV().col(8);
#else
        Matx<double, 9, 9> AtA_(AtA), U, Vt;
        Vec<double, 9> W;
        SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
        models = std::vector<Mat> { Mat_<double>(3, 3, Vt.val + 72 /*=8*9*/) };
#endif
        if (enforce_rank)
            FundamentalDegeneracy::recoverRank(models[0], is_fundamental);
        if (is_fundamental) {
            const auto * const f = (double *) models[0].data;
            auto * t1 = normTr ? T1.val : _T1.val, * t2 = normTr ? T2.val : _T2.val;

            // F = T2^T F T1
            models[0] = Mat(Matx33d(t1[0]*t2[0]*f[0],t1[0]*t2[0]*f[1], t2[0]*f[2] + t2[0]*f[0]*t1[2] +
                t2[0]*f[1]*t1[5], t1[0]*t2[0]*f[3],t1[0]*t2[0]*f[4], t2[0]*f[5] + t2[0]*f[3]*t1[2] +
                t2[0]*f[4]*t1[5], t1[0]*(f[6] + f[0]*t2[2] + f[3]*t2[5]), t1[0]*(f[7] + f[1]*t2[2] +
                f[4]*t2[5]), f[8] + t1[2]*(f[6] + f[0]*t2[2] + f[3]*t2[5]) + t1[5]*(f[7] + f[1]*t2[2] +
                f[4]*t2[5]) + f[2]*t2[2] + f[5]*t2[5]));
        }
        return 1;
    }

    int getMinimumRequiredSampleSize() const override { return 8; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<EpipolarNonMinimalSolver> EpipolarNonMinimalSolver::create(const Mat &points_, bool is_fundamental) {
    return makePtr<EpipolarNonMinimalSolverImpl>(points_, is_fundamental);
}
Ptr<EpipolarNonMinimalSolver> EpipolarNonMinimalSolver::create(const Mat &points_, const Matx33d &T1, const Matx33d &T2) {
    return makePtr<EpipolarNonMinimalSolverImpl>(points_, T1, T2);
}

}}
