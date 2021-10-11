#include "precomp.hpp"

namespace cv { namespace vsac {
class HomographyMinimalSolver4ptsGEMImpl : public HomographyMinimalSolver4ptsGEM {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit HomographyMinimalSolver4ptsGEMImpl (const Mat &points_) :
        points_mat(&points_), points ((float*) points_.data) {}

    int estimate (const std::vector<int>& sample, std::vector<Mat> &models) const override {
        int m = 8, n = 9;
        std::vector<double> A(72, 0);
        int cnt = 0;
        for (int i = 0; i < 4; i++) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl], y1 = points[smpl+1], x2 = points[smpl+2], y2 = points[smpl+3];

            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            cnt += 3; // skip zeros
            A[cnt++] = x2*x1;
            A[cnt++] = x2*y1;
            A[cnt++] = x2;

            cnt += 3;
            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            A[cnt++] = y2*x1;
            A[cnt++] = y2*y1;
            A[cnt++] = y2;
        }

        if (!Math::eliminateUpperTriangular(A, m, n))
            return 0;

        models = std::vector<Mat>{ Mat_<double>(3,3) };
        auto * h = (double *) models[0].data;
        h[8] = 1.;

        // start from the last row
        for (int i = m-1; i >= 0; i--) {
            double acc = 0;
            for (int j = i+1; j < n; j++)
                acc -= A[i*n+j]*h[j];

            h[i] = acc / A[i*n+i];
            // due to numerical errors return 0 solutions
            if (std::isnan(h[i]))
                return 0;
        }
        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 4; }
};
Ptr<HomographyMinimalSolver4ptsGEM> HomographyMinimalSolver4ptsGEM::create(const Mat &points_) {
    return makePtr<HomographyMinimalSolver4ptsGEMImpl>(points_);
}

class HomographySVDSolverImpl: public HomographySVDSolver {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit HomographySVDSolverImpl (const Mat &points_) :
            points_mat (&points_), points ((float *) points_.data) {}

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        cv::Mat A_svd = cv::Mat_<double>::zeros(8, 9);
        auto * A = (double *) A_svd.data;
        int cnt = 0;
        for (int i = 0; i < 4; i++ ) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl  ], y1 = points[smpl+1],
                       x2 = points[smpl+2], y2 = points[smpl+3];

            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            cnt += 3; // skip zeros
            A[cnt++] = x2*x1;
            A[cnt++] = x2*y1;
            A[cnt++] = x2;

            cnt += 3;
            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            A[cnt++] = y2*x1;
            A[cnt++] = y2*y1;
            A[cnt++] = y2;
        }

        Mat U, Vt, D;
        SVD::compute(A_svd, D, U, Vt, SVD::FULL_UV+SVD::MODIFY_A);

        models = std::vector<Mat> { Vt.row(Vt.rows-1).reshape(0, 3) };
        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 4; }
};
Ptr<HomographySVDSolver> HomographySVDSolver::create(const Mat &points_) {
    return makePtr<HomographySVDSolverImpl>(points_);
}


class HomographyNonMinimalSolverImpl : public HomographyNonMinimalSolver {
private:
    const Mat * points_mat;
    Ptr<NormTransform> normTr = nullptr;
    Matx33d _T1, _T2;
public:
    explicit HomographyNonMinimalSolverImpl (const Mat &norm_points_, const Matx33d &T1, const Matx33d &T2) :
            points_mat(&norm_points_), _T1(T1), _T2(T2) {}
    explicit HomographyNonMinimalSolverImpl (const Mat &points_) :
        points_mat(&points_), normTr (NormTransform::create(points_)) {}

    /*
     * Find Homography matrix using (weighted) non-minimal estimation.
     * Use Principal Component Analysis. Use normalized points.
     */
    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Matx33d T1, T2;
        Mat norm_points_;
        if (normTr)
            normTr->getNormTransformation(norm_points_, sample, sample_size, T1, T2);
        const auto * const norm_points = normTr ? (float *) norm_points_.data : (float *) points_mat->data;

        double a1[9] = {0, 0, -1, 0, 0, 0, 0, 0, 0},
               a2[9] = {0, 0, 0, 0, 0, -1, 0, 0, 0},
               AtA[81] = {0};

        if (weights.empty()) {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const auto x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
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

                for (int j = 0; j < 9; j++)
                    for (int z = j; z < 9; z++)
                        AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        } else {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const double weight = weights[i];
                const auto x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                           x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];
                const double minus_weight_times_x1 = -weight * x1,
                             minus_weight_times_y1 = -weight * y1,
                                   weight_times_x2 =  weight * x2,
                                   weight_times_y2 =  weight * y2;

                a1[0] = minus_weight_times_x1;
                a1[1] = minus_weight_times_y1;
                a1[2] = -weight;
                a1[6] = weight_times_x2 * x1;
                a1[7] = weight_times_x2 * y1;
                a1[8] = weight_times_x2;

                a2[3] = minus_weight_times_x1;
                a2[4] = minus_weight_times_y1;
                a2[5] = -weight;
                a2[6] = weight_times_y2 * x1;
                a2[7] = weight_times_y2 * y1;
                a2[8] = weight_times_y2;

                for (int j = 0; j < 9; j++)
                    for (int z = j; z < 9; z++)
                        AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        }

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                AtA[j*9+z] = AtA[z*9+j];

#ifdef HAVE_EIGEN
        Mat H = Mat_<double>(3,3);
        // extract the last null-vector
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Eigen::Matrix<double, 9, 9>
                (Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> (
                        (Eigen::Matrix<double, 9, 9> (AtA))).householderQ()).col(8);
#else
        Matx<double, 9, 9> Vt;
        Vec<double, 9> D;
        if (! eigen(Matx<double, 9, 9>(AtA), D, Vt)) return 0;
        Mat H = Mat_<double>(3, 3, Vt.val + 72/*=8*9*/);
#endif
        const auto * const h = (double *) H.data;
        const auto * const t1 = normTr ? T1.val : _T1.val, * const t2 = normTr ? T2.val : _T2.val;
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

    int getMinimumRequiredSampleSize() const override { return 4; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<HomographyNonMinimalSolver> HomographyNonMinimalSolver::create(const Mat &points_) {
    return makePtr<HomographyNonMinimalSolverImpl>(points_);
}
Ptr<HomographyNonMinimalSolver> HomographyNonMinimalSolver::create(const Mat &points_, const Matx33d &T1, const Matx33d &T2) {
    return makePtr<HomographyNonMinimalSolverImpl>(points_, T1, T2);
}

class AffineMinimalSolverImpl : public AffineMinimalSolver {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit AffineMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_), points((float *) points_.data) {}
    /*
        Affine transformation
        x1 y1 1 0  0  0   a   u1
        0  0  0 x1 y1 1   b   v1
        x2 y2 1 0  0  0   c   u2
        0  0  0 x2 y2 1 * d = v2
        x3 y3 1 0  0  0   e   u3
        0  0  0 x3 y3 1   f   v3
    */
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2];
        const auto
                x1 = points[smpl1], y1 = points[smpl1+1], u1 = points[smpl1+2], v1 = points[smpl1+3],
                x2 = points[smpl2], y2 = points[smpl2+1], u2 = points[smpl2+2], v2 = points[smpl2+3],
                x3 = points[smpl3], y3 = points[smpl3+1], u3 = points[smpl3+2], v3 = points[smpl3+3];

        // covers degeneracy test when all 3 points are collinear.
        // In this case denominator will be 0
        double denominator = x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2;
        if (fabs(denominator) < FLT_EPSILON) // check if denominator is zero
            return 0;
        denominator = 1. / denominator;

        double a =  (u1*y2 - u2*y1 - u1*y3 + u3*y1 + u2*y3 - u3*y2) * denominator;
        double b = -(u1*x2 - u2*x1 - u1*x3 + u3*x1 + u2*x3 - u3*x2) * denominator;
        double c = u1 - a * x1 - b * y1; // ax1 + by1 + c = u1
        double d =  (v1*y2 - v2*y1 - v1*y3 + v3*y1 + v2*y3 - v3*y2) * denominator;
        double e = -(v1*x2 - v2*x1 - v1*x3 + v3*x1 + v2*x3 - v3*x2) * denominator;
        double f = v1 - d * x1 - e * y1; // dx1 + ey1 + f = v1

        models[0] = Mat(Matx33d(a, b, c, d, e, f, 0, 0, 1));
        return 1;
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<AffineMinimalSolver> AffineMinimalSolver::create(const Mat &points_) {
    return makePtr<AffineMinimalSolverImpl>(points_);
}

class AffineNonMinimalSolverImpl : public AffineNonMinimalSolver {
private:
    const Mat * points_mat;
    Ptr<NormTransform> normTr = nullptr;
    Matx33d _T1, _T2;
public:
    explicit AffineNonMinimalSolverImpl (const Mat &points_, InputArray T1, InputArray T2) :
            points_mat(&points_) {
        if (!T1.empty() && !T2.empty()) {
            normTr = NormTransform::create(points_);
            _T1 = T1.getMat();
            _T2 = T2.getMat();
        }
    }

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;
        Matx33d T1, T2;
        Mat norm_points_;
        if (normTr)
            normTr->getNormTransformation(norm_points_, sample, sample_size, T1, T2);
        const auto * const pts = normTr ? (float *) norm_points_.data : (float *) points_mat->data;

        // do Least Squares
        // Ax = b   ->  A^T Ax = A^T b
        // x = (A^T A)^-1 A^T b
        double AtA[36] = {0}, Ab[6] = {0};
        double r1[6] = {0, 0, 1, 0, 0, 0}; // row 1 of A
        double r2[6] = {0, 0, 0, 0, 0, 1}; // row 2 of A

        if (weights.empty())
            for (int p = 0; p < sample_size; p++) {
                // if (weights != nullptr) weight = weights[sample[p]];

                const int smpl = 4*sample[p];
                const double x1=pts[smpl], y1=pts[smpl+1], x2=pts[smpl+2], y2=pts[smpl+3];

                r1[0] = x1;
                r1[1] = y1;

                r2[3] = x1;
                r2[4] = y1;

                for (int j = 0; j < 6; j++) {
                    for (int z = j; z < 6; z++)
                        AtA[j * 6 + z] += r1[j] * r1[z] + r2[j] * r2[z];
                    Ab[j] += r1[j]*x2 + r2[j]*y2;
                }
            }
        else
            for (int p = 0; p < sample_size; p++) {
                const int smpl = 4*sample[p];
                const double weight = weights[p];
                const double weight_times_x1 = weight * pts[smpl  ],
                             weight_times_y1 = weight * pts[smpl+1],
                             weight_times_x2 = weight * pts[smpl+2],
                             weight_times_y2 = weight * pts[smpl+3];

                r1[0] = weight_times_x1;
                r1[1] = weight_times_y1;
                r1[2] = weight;

                r2[3] = weight_times_x1;
                r2[4] = weight_times_y1;
                r2[5] = weight;

                for (int j = 0; j < 6; j++) {
                    for (int z = j; z < 6; z++)
                        AtA[j * 6 + z] += r1[j] * r1[z] + r2[j] * r2[z];
                    Ab[j] += r1[j]*weight_times_x2 + r2[j]*weight_times_y2;
                }
            }

        // copy symmetric part
        for (int j = 1; j < 6; j++)
            for (int z = 0; z < j; z++)
                AtA[j*6+z] = AtA[z*6+j];

        Vec6d aff;
        if (!solve(Matx66d(AtA), Vec6d(Ab), aff))
            return 0;
        const double h[9] = {aff(0), aff(1), aff(2),
                             aff(3), aff(4), aff(5),
                             0, 0, 1};
        const auto * const t1 = normTr ? T1.val : _T1.val, * const t2 = normTr ? T2.val : _T2.val;
        // A = T2^-1 A T1
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(h[0]/t2[0] - (h[6]*t2[2])/t2[0]),
                t1[0]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]), h[2]/t2[0] + t1[2]*(h[0]/t2[0] -
                (h[6]*t2[2])/t2[0]) + t1[5]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]) - (h[8]*t2[2])/t2[0],
                t1[0]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]), t1[0]*(h[4]/t2[0] - (h[7]*t2[5])/t2[0]),
                h[5]/t2[0] + t1[2]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]) + t1[5]*(h[4]/t2[0] -
                (h[7]*t2[5])/t2[0]) - (h[8]*t2[5])/t2[0], t1[0]*h[6], t1[0]*h[7],
                h[8] + h[6]*t1[2] + h[7]*t1[5])) };

        // models[0] = T2.inv() * models[0] * T1;
        return 1;
    }

    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<AffineNonMinimalSolver> AffineNonMinimalSolver::create(const Mat &points_, InputArray T1, InputArray T2) {
    return makePtr<AffineNonMinimalSolverImpl>(points_, T1, T2);
}
}}