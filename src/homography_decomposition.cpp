// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv { namespace vsac {
double oppositeOfMinor(const Matx33d& M, const int row, const int col);
void findRmatFrom_tstar_n(const Matx33d &Hnorm, const cv::Vec3d& tstar, const cv::Vec3d& n, const double v, cv::Matx33d& R);

// OpenCV:
double oppositeOfMinor(const Matx33d& M, const int row, const int col) {
    const int x1 = col == 0 ? 1 : 0, x2 = col == 2 ? 1 : 2;
    const int y1 = row == 0 ? 1 : 0, y2 = row == 2 ? 1 : 2;
    return (M(y1, x2) * M(y2, x1) - M(y1, x1) * M(y2, x2));
}

void findRmatFrom_tstar_n(const Matx33d &Hnorm, const cv::Vec3d& tstar, const cv::Vec3d& n, const double v, cv::Matx33d& R) {
    R = Hnorm * (Matx33d::eye() - (2/v) * Matx31d(tstar) *  Matx31d(n).t());
    if (determinant(R) < 0)
        R *= -1;
}
void Utils::getClosePoints (const cv::Mat &points, std::vector<std::vector<int>> &close_points, double close_thr_sqr) {
    const auto close_thr = sqrtf((float)close_thr_sqr);
    const auto graph = cv::vsac::GridNeighborhoodGraph2::create(points, points.rows, close_thr, close_thr, close_thr, close_thr);
    close_points = graph->getGraph();
}
/*
 * Hnorm = K2^-1 H K1 -- normalized homography
 */
int Utils::decomposeHomography (const Matx33d &Hnorm_, std::vector<Matx33d> &R, std::vector<Vec3d> &t) {
    // remove scale
    Vec3d w;
    SVD::compute(Hnorm_, w);
    Matx33d Hnorm = Hnorm_ * (1/w(1));

    const double epsilon = 0.003;
    //S = H'H - I
    Matx33d S = Hnorm.t() * Hnorm;
    S(0, 0) -= 1.0;
    S(1, 1) -= 1.0;
    S(2, 2) -= 1.0;

    //check if H is rotation matrix
    if (norm(S, NORM_INF) < epsilon) {
        R = std::vector<Matx33d> { Hnorm };
        t = std::vector<Vec3d> { Vec3d(0,0,0) };
        return 1;
    }

    //! Compute nvectors
    const double M00 = oppositeOfMinor(S, 0, 0);
    const double M11 = oppositeOfMinor(S, 1, 1);
    const double M22 = oppositeOfMinor(S, 2, 2);

    const double rtM00 = sqrt(M00);
    const double rtM11 = sqrt(M11);
    const double rtM22 = sqrt(M22);

    const double M01 = oppositeOfMinor(S, 0, 1);
    const double M12 = oppositeOfMinor(S, 1, 2);
    const double M02 = oppositeOfMinor(S, 0, 2);

    const auto signd = [] (double x) { return x >= 0 ? 1 : -1; };

    const int e12 = signd(M12);
    const int e02 = signd(M02);
    const int e01 = signd(M01);

    const double nS00 = abs(S(0, 0));
    const double nS11 = abs(S(1, 1));
    const double nS22 = abs(S(2, 2));

    //find max( |Sii| ), i=0, 1, 2
    int indx = 0;
    if (nS00 < nS11){
        indx = 1;
        if( nS11 < nS22 )
            indx = 2;
    } else {
        if(nS00 < nS22 )
            indx = 2;
    }

    Vec3d npa, npb;
    switch (indx) {
        case 0:
            npa[0] = S(0, 0),               npb[0] = S(0, 0);
            npa[1] = S(0, 1) + rtM22,       npb[1] = S(0, 1) - rtM22;
            npa[2] = S(0, 2) + e12 * rtM11, npb[2] = S(0, 2) - e12 * rtM11;
            break;
        case 1:
            npa[0] = S(0, 1) + rtM22,       npb[0] = S(0, 1) - rtM22;
            npa[1] = S(1, 1),               npb[1] = S(1, 1);
            npa[2] = S(1, 2) - e02 * rtM00, npb[2] = S(1, 2) + e02 * rtM00;
            break;
        case 2:
            npa[0] = S(0, 2) + e01 * rtM11, npb[0] = S(0, 2) - e01 * rtM11;
            npa[1] = S(1, 2) + rtM00,       npb[1] = S(1, 2) - rtM00;
            npa[2] = S(2, 2),               npb[2] = S(2, 2);
            break;
        default:
            break;
    }

    const double traceS = S(0, 0) + S(1, 1) + S(2, 2);
    const double v = 2.0 * sqrt(1 + traceS - M00 - M11 - M22);
    const double n_t = sqrt(2 + traceS - v);
    const double half_nt = 0.5 * n_t;
    const double esii_t_r = signd(S(indx, indx)) * sqrt(2 + traceS + v);

    const Vec3d na = npa / norm(npa);
    const Vec3d nb = npb / norm(npb);

    const Vec3d ta_star = half_nt * (esii_t_r * nb - n_t * na);
    const Vec3d tb_star = half_nt * (esii_t_r * na - n_t * nb);

    //Ra=R1, ta=t1, na
    Matx33d R1, R2;
    findRmatFrom_tstar_n(Hnorm, ta_star, na, v, R1);
    //Rb=R2, tb=t2, nb
    findRmatFrom_tstar_n(Hnorm, tb_star, nb, v, R2);
    R = {R1, R2};
    t = {R1 * ta_star, R2 * tb_star};
    return 2;
}
}}