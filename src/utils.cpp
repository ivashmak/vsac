#include "precomp.hpp"
#include <opencv2/flann/miniflann.hpp>

namespace cv { namespace vsac {
double Utils::getCalibratedThreshold (double threshold, const Mat &K1, const Mat &K2) {
    const auto * const k1 = (double *) K1.data, * const k2 = (double *) K2.data;
    return threshold / ((k1[0] + k1[4] + k2[0] + k2[4]) / 4.0);
}

/*
 * K1, K2 are 3x3 intrinsics matrices
 * points is matrix of size |N| x 4
 * Assume K = [k11 k12 k13
 *              0  k22 k23
 *              0   0   1]
 */
void Utils::calibratePoints (const Mat &K1, const Mat &K2, const Mat &points, Mat &calib_points) {
    const auto * const points_ = (float *) points.data;
    const auto * const k1 = (double *) K1.data;
    const auto inv1_k11 = float(1 / k1[0]); // 1 / k11
    const auto inv1_k12 = float(-k1[1] / (k1[0]*k1[4])); // -k12 / (k11*k22)
    // (-k13*k22 + k12*k23) / (k11*k22)
    const auto inv1_k13 = float((-k1[2]*k1[4] + k1[1]*k1[5]) / (k1[0]*k1[4]));
    const auto inv1_k22 = float(1 / k1[4]); // 1 / k22
    const auto inv1_k23 = float(-k1[5] / k1[4]); // -k23 / k22

    const auto * const k2 = (double *) K2.data;
    const auto inv2_k11 = float(1 / k2[0]);
    const auto inv2_k12 = float(-k2[1] / (k2[0]*k2[4]));
    const auto inv2_k13 = float((-k2[2]*k2[4] + k2[1]*k2[5]) / (k2[0]*k2[4]));
    const auto inv2_k22 = float(1 / k2[4]);
    const auto inv2_k23 = float(-k2[5] / k2[4]);

    calib_points = Mat ( points.rows, 4, points.type());
    auto * calib_points_ = (float *) calib_points.data;

    for (int i = 0; i <  points.rows; i++) {
        const int idx = 4*i;
        (*calib_points_++) = inv1_k11 * points_[idx  ] + inv1_k12 * points_[idx+1] + inv1_k13;
        (*calib_points_++) =                             inv1_k22 * points_[idx+1] + inv1_k23;
        (*calib_points_++) = inv2_k11 * points_[idx+2] + inv2_k12 * points_[idx+3] + inv2_k13;
        (*calib_points_++) =                             inv2_k22 * points_[idx+3] + inv2_k23;
    }
}

/*
 * K is 3x3 intrinsic matrix
 * points is matrix of size |N| x 5, first two columns are image points [u_i, v_i]
 * calib_norm_pts are  K^-1 [u v 1]^T / ||K^-1 [u v 1]^T||
 */
void Utils::calibrateAndNormalizePointsPnP (const Mat &K, const Mat &pts, Mat &calib_norm_pts) {
    const auto * const points = (float *) pts.data;
    const auto * const k = (double *) K.data;
    const auto inv_k11 = float(1 / k[0]);
    const auto inv_k12 = float(-k[1] / (k[0]*k[4]));
    const auto inv_k13 = float((-k[2]*k[4] + k[1]*k[5]) / (k[0]*k[4]));
    const auto inv_k22 = float(1 / k[4]);
    const auto inv_k23 = float(-k[5] / k[4]);

    calib_norm_pts = Mat (pts.rows, 3, pts.type());
    auto * calib_norm_pts_ = (float *) calib_norm_pts.data;

    for (int i = 0; i < pts.rows; i++) {
        const int idx = 5 * i;
        const float k_inv_u = inv_k11 * points[idx] + inv_k12 * points[idx+1] + inv_k13;
        const float k_inv_v =                         inv_k22 * points[idx+1] + inv_k23;
        const float norm = 1.f / sqrtf(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        (*calib_norm_pts_++) = k_inv_u * norm;
        (*calib_norm_pts_++) = k_inv_v * norm;
        (*calib_norm_pts_++) =           norm;
    }
}

void Utils::normalizeAndDecalibPointsPnP (const Mat &K_, Mat &pts, Mat &calib_norm_pts) {
    const auto * const K = (double *) K_.data;
    const auto k11 = (float)K[0], k12 = (float)K[1], k13 = (float)K[2],
               k22 = (float)K[4], k23 = (float)K[5];
    calib_norm_pts = Mat (pts.rows, 3, pts.type());
    auto * points = (float *) pts.data;
    auto * calib_norm_pts_ = (float *) calib_norm_pts.data;

    for (int i = 0; i < pts.rows; i++) {
        const int idx = 5 * i;
        const float k_inv_u = points[idx  ];
        const float k_inv_v = points[idx+1];
        const float norm = 1.f / sqrtf(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        (*calib_norm_pts_++) = k_inv_u * norm;
        (*calib_norm_pts_++) = k_inv_v * norm;
        (*calib_norm_pts_++) =           norm;
        points[idx  ] = k11 * k_inv_u + k12 * k_inv_v + k13;
        points[idx+1] =                 k22 * k_inv_v + k23;
    }
}
/*
 * decompose Projection Matrix to calibration, rotation and translation
 * Assume K = [fx  0   tx
 *             0   fy  ty
 *             0   0   1]
 */
void Utils::decomposeProjection (const Mat &P, Mat &K_, Mat &R, Mat &t, bool same_focal) {
    const Mat M = P.colRange(0,3);
    double scale = norm(M.row(2)); scale *= scale;
    Matx33d K = Matx33d::eye();
    K(1,2) = M.row(1).dot(M.row(2)) / scale;
    K(0,2) = M.row(0).dot(M.row(2)) / scale;
    K(1,1) = sqrt(M.row(1).dot(M.row(1)) / scale - K(1,2)*K(1,2));
    K(0,0) = sqrt(M.row(0).dot(M.row(0)) / scale - K(0,2)*K(0,2));
    if (same_focal)
        K(0,0) = K(1,1) = (K(0,0) + K(1,1)) / 2;
    R = K.inv() * M / sqrt(scale);
    if (determinant(M) < 0) R *= -1;
    t = R * M.inv() * P.col(3);
    K_ = Mat(K);
}

void Utils::densitySort (const Mat &points, int knn, Mat &sorted_points, std::vector<int> &sorted_mask) {
    // mask of sorted points (array of indexes)
    const int points_size = points.rows, dim = points.cols;
    sorted_mask = std::vector<int >(points_size);
    for (int i = 0; i < points_size; i++)
        sorted_mask[i] = i;

    // get neighbors
    FlannNeighborhoodGraph &graph = *FlannNeighborhoodGraph::create(points, points_size, knn,
            true /*get distances */, 6, 1);

    std::vector<double> sum_knn_distances (points_size, 0);
    for (int p = 0; p < points_size; p++) {
        const std::vector<double> &dists = graph.getNeighborsDistances(p);
        for (int k = 0; k < knn; k++)
            sum_knn_distances[p] += dists[k];
    }

    // compare by sum of distances to k nearest neighbors.
    std::sort(sorted_mask.begin(), sorted_mask.end(), [&](int a, int b) {
        return sum_knn_distances[a] < sum_knn_distances[b];
    });

    // copy array of points to array with sorted points
    // using @sorted_idx mask of sorted points indexes

    sorted_points = Mat(points_size, dim, points.type());
    const auto * const points_ptr = (float *) points.data;
    auto * spoints_ptr = (float *) sorted_points.data;
    for (int i = 0; i < points_size; i++) {
        const int pt2 = sorted_mask[i] * dim;
        for (int j = 0; j < dim; j++)
            (*spoints_ptr++) =  points_ptr[pt2+j];
    }
}

Matx33d Math::getSkewSymmetric(const Vec3d &v) {
     return Matx33d(0,    -v[2], v[1],
                   v[2],  0,    -v[0],
                  -v[1],  v[0], 0);
}

Matx33d Math::rotVec2RotMat (const Vec3d &v) {
    const double phi = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    const double x = v[0] / phi, y = v[1] / phi, z = v[2] / phi;
    const double a = sin(phi), b = cos(phi);
    // R = I + sin(phi) * skew(v) + (1 - cos(phi) * skew(v)^2
    return Matx33d((b - 1)*y*y + (b - 1)*z*z + 1, -a*z - x*y*(b - 1), a*y - x*z*(b - 1),
     a*z - x*y*(b - 1), (b - 1)*x*x + (b - 1)*z*z + 1, -a*x - y*z*(b - 1),
    -a*y - x*z*(b - 1), a*x - y*z*(b - 1), (b - 1)*x*x + (b - 1)*y*y + 1);
}

Vec3d Math::rotMat2RotVec (const Matx33d &R) {
    // https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio?rq=1
    Vec3d rot_vec;
    const double trace = R(0,0)+R(1,1)+R(2,2);
    if (trace >= 3 - FLT_EPSILON) {
        rot_vec = (0.5 * (trace-3)/12)*Vec3d(R(2,1)-R(1,2),
                                             R(0,2)-R(2,0),
                                             R(1,0)-R(0,1));
    } else if (3 - FLT_EPSILON > trace && trace > -1 + FLT_EPSILON) {
        double theta = acos((trace - 1) / 2);
        rot_vec = (theta / (2 * sin(theta))) * Vec3d(R(2,1)-R(1,2),
                                                     R(0,2)-R(2,0),
                                                     R(1,0)-R(0,1));
    } else {
        int a;
        if (R(0,0) > R(1,1))
            a = R(0,0) > R(2,2) ? 0 : 2;
        else
            a = R(1,1) > R(2,2) ? 1 : 2;
        Vec3d v;
        int b = (a + 1) % 3, c = (a + 2) % 3;
        double s = sqrt(R(a,a) - R(b,b) - R(c,c) + 1);
        v[a] = s / 2;
        v[b] = (R(b,a) + R(a,b)) / (2 * s);
        v[c] = (R(c,a) + R(a,c)) / (2 * s);
        rot_vec = M_PI * v / norm(v);
    }
    return rot_vec;
}

/*
 * Eliminate matrix of m rows and n columns to be upper triangular.
 */
bool Math::eliminateUpperTriangular (std::vector<double> &a, int m, int n) {
    for (int r = 0; r < m; r++){
        double pivot = a[r*n+r];
        int row_with_pivot = r;

        // find the maximum pivot value among r-th column
        for (int k = r+1; k < m; k++)
            if (fabs(pivot) < fabs(a[k*n+r])) {
                pivot = a[k*n+r];
                row_with_pivot = k;
            }

        // if pivot value is 0 continue
        if (fabs(pivot) < DBL_EPSILON) {
            continue;
            // return false; // matrix is not full rank -> terminate
        }

        // swap row with maximum pivot value with current row
        for (int c = r; c < n; c++)
            std::swap(a[row_with_pivot*n+c], a[r*n+c]);

        // eliminate other rows
        for (int j = r+1; j < m; j++){
            const int row_idx1 = j*n, row_idx2 = r*n;
            const auto fac = a[row_idx1+r] / pivot;
            a[row_idx1+r] = 0; // zero eliminated element
            for (int c = r+1; c < n; c++)
                a[row_idx1+c] -= fac * a[row_idx2+c];
        }
    }
    return true;
}

double Utils::intersectionOverUnion (const std::vector<bool> &a, const std::vector<bool> &b) {
    int intersects = 0, unions = 0;
    for (int i = 0; i < (int)a.size(); i++)
        if (a[i] || b[i]) {
            unions++; // one value is true
            if (a[i] && b[i])
                intersects++; // a[i] == b[i] and if they both true
        }
    return (double) intersects / unions;
}

double Utils::getPoissonCDF (double lambda, int inliers) {
    double exp_lamda = exp(-lambda), cdf = exp_lamda, lambda_i_div_fact_i = 1;
    for (int i = 1; i <= inliers; i++) {
        lambda_i_div_fact_i *= (lambda / i);
        cdf += exp_lamda * lambda_i_div_fact_i;
        if (fabs(cdf - 1) < DBL_EPSILON) // cdf is almost 1
            break;
    }
    return cdf;
}

// OpenCV:
double oppositeOfMinor(const Matx33d& M, const int row, const int col);
int signd (double x);
void findRmatFrom_tstar_n(const Matx33d &Hnorm, const cv::Vec3d& tstar, const cv::Vec3d& n, const double v, cv::Matx33d& R);

// OpenCV:
double oppositeOfMinor(const Matx33d& M, const int row, const int col) {
    const int x1 = col == 0 ? 1 : 0, x2 = col == 2 ? 1 : 2;
    const int y1 = row == 0 ? 1 : 0, y2 = row == 2 ? 1 : 2;
    return (M(y1, x2) * M(y2, x1) - M(y1, x1) * M(y2, x2));
}
int signd (double x) {
    return x >= 0 ? 1 : -1;
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

    // std::cout << "norm error " << norm(S, NORM_INF) << "\n";
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
int Utils::triangulatePointsRt (const Mat &points, Mat &points3D, const Mat &K1_, const Mat &K2_, 
        const cv::Mat &R, const cv::Mat &t_vec, std::vector<bool> &good_mask, std::vector<double> &depths1, std::vector<double> &depths2) {
    cv::Matx33d K1 = Matx33d(K1_), K2 = Matx33d(K2_);
    cv::Matx34d P2;
    cv::hconcat(K2 * R, K2 * t_vec, P2);
    cv::Matx66d A = cv::Matx66d::zeros();
    A(2,0) = A(5,1) = 1;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            A(  r,2+c) = -K1   (r,c);
            A(3+r,2+c) = -P2(r,c);
        }
    }
    good_mask = std::vector<bool> (points.rows, false);
    depths1 = std::vector<double>(points.rows,0);
    depths2 = std::vector<double>(points.rows,0);
    points3D = Mat_<float>::zeros(points.rows, 3);
    const auto * const pts = (float *) points.data;
    auto * pts3D_ptr = (float *) points3D.data;
    int num_valid_pts = 0;
    for (int i = 0; i < points.rows; i++) {
        A(0,0) = pts[4*i  ];
        A(1,0) = pts[4*i+1];
        A(3,1) = pts[4*i+2];
        A(4,1) = pts[4*i+3];
        // https://cw.fel.cvut.cz/wiki/_media/courses/gvg/pajdla-gvg-lecture-2021.pdf
        cv::Matx66d U, Vt;
        cv::Vec6d D;
        cv::SVDecomp(A, D, U, Vt);
        const double scale1 = Vt(5,0) / Vt(5,5), scale2 = Vt(5,1) / Vt(5,5);
            
        // cv::Vec4d X;
        // cv::Vec2d pt1(pts[4*i],pts[4*i+1]), pt2(pts[4*i+2],pts[4*i+3]);
        // cv::triangulatePoints(P1, P2[p], pt1, pt2, X);
        // const auto x1 = P1 * X, x2 = P2[p] * X;
        // since P1 = K [I | 0] it imples that if scale1 < 0 then z < 0
        if (scale1 > 0 && scale2 > 0) {
            // std::cout << "good pt " << i << " scales " << scale1 << " " << scale2 << '\n';
            good_mask[i] = true;
            pts3D_ptr[i*3  ] = Vt(5,2) / Vt(5,5);
            pts3D_ptr[i*3+1] = Vt(5,3) / Vt(5,5);
            pts3D_ptr[i*3+2] = Vt(5,4) / Vt(5,5);
            depths1[i] = scale1;
            depths2[i] = scale2;
            num_valid_pts++;
        }
    }
    return num_valid_pts;
}
void Utils::triangulatePoints (const Mat &points, const Mat &E, const Mat &K1_, const Mat &K2_, Mat &points3D, 
        cv::Mat &R, cv::Mat &t_vec, std::vector<bool> &good_mask, std::vector<double> &depths2, std::vector<double> &depths1) {
    
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);

    std::vector<Mat> pts3D(4);
    std::vector<std::vector<bool>> good_masks(4);
    std::vector<std::vector<double>> depths2_(4), depths1_(4);
    int pts_in_front[4] = {
        Utils::triangulatePointsRt(points, pts3D[0], K1_, K2_, R1,  t, good_masks[0], depths1_[0], depths2_[0]),
        Utils::triangulatePointsRt(points, pts3D[1], K1_, K2_, R1, -t, good_masks[1], depths1_[1], depths2_[1]),
        Utils::triangulatePointsRt(points, pts3D[2], K1_, K2_, R2,  t, good_masks[2], depths1_[2], depths2_[2]),
        Utils::triangulatePointsRt(points, pts3D[3], K1_, K2_, R2, -t, good_masks[3], depths1_[3], depths2_[3])  
    };

    int max_points_in_front = 0, correct_idx = 0;
    for (int p = 0; p < 4; p++)
        if (max_points_in_front < pts_in_front[p]) {
            max_points_in_front = pts_in_front[p];
            correct_idx = p;
        }

    if (correct_idx <= 1)
         R = Mat(R1);
    else R = Mat(R2);
    if (correct_idx % 2)
         t_vec = Mat(-t);
    else t_vec = Mat( t);
    pts3D[correct_idx].copyTo(points3D);
    good_mask = good_masks[correct_idx];
    depths1 = depths1_[correct_idx];
    depths2 = depths2_[correct_idx];
}

void Utils::triangulatePoints (bool calibrated, const Mat &points, const Mat &E_, Mat &new_points, Mat &points3D, Mat &R) {
    /*
     * S = (1 0 0)
     *      0 1 0)
     */
    auto sgn = [] (const double v) {
        return (0 < v) - (v < 0);
    };

    new_points = Mat(points.rows, points.cols, points.type());
    int num_pts_positive_depth[2] = {0};

    cv::Mat points3D_R1, points3D_R2;
    float * pts3D_R1, * pts3D_R2;
    if (calibrated) {
        points3D_R1 = Mat_<float>(points.rows, 3);
        points3D_R2 = Mat_<float>(points.rows, 3);
        pts3D_R1 = (float *) points3D_R1.data;
        pts3D_R2 = (float *) points3D_R2.data;
    }
    auto * new_pts = (float *) new_points.data;

    Vec3d t;
    Matx33d R1, R2, E (E_);
    if (calibrated)
        decomposeEssentialMat(E_, R1, R2, t);
    E = E.t();
    const Matx23d S (1, 0, 0, 0, 1, 0);
    const Matx32d St = S.t();
    const Matx23d SE = S * E, SEt = S * E.t();
    const Matx22d SESt = S * E * St;
    const auto * const pts = (float *) points.data;

    for (int pt = 0; pt < points.rows; pt++) {
        Vec3d x (pts[4*pt], pts[4*pt+1], 1);
        Vec3d xp (pts[4*pt+2], pts[4*pt+3], 1);

       // std::cout << "before " << (xp.t() * E * x) << " " << (x.t() * E * xp) << " x " << x << " xp " << xp << "\n";

//        Vec3d xp (pts[4*pt], pts[4*pt+1], 1);
//        Vec3d x (pts[4*pt+2], pts[4*pt+3], 1);

        auto n = SE * xp, np = SEt * x;
//        const auto a = (n[0] * SESt(0,0) + n[1] * SESt(1,0)) * np[0] +
//                       (n[0] * SESt(0,1) + n[1] * SESt(1,1)) * np[1];
        const auto a = (Vec2d(Mat(n.t() * SESt))).dot(np);
        const auto b = 0.5 * (n.dot(n) + np.dot(np));
//        const auto c = (x[0] * E(0,0) + x[1] * E(1,0) + E(2,0)) * xp[0] +
//                       (x[0] * E(0,1) + x[1] * E(1,1) + E(2,1)) * xp[1] +
//                       (x[0] * E(0,2) + x[1] * E(1,2) + E(2,2)); // xT E x
        const auto c = (Vec3d(Mat(x.t() * E))).dot(xp);
        auto lambda = c / (b + sqrt(b*b - a * c));
        auto dx = lambda * n;
        auto dxp = lambda * np;
        n -= SESt * dxp;
        np -= SESt.t() * dx;

        // niter1
        dx = dx.dot(n) * n / n.dot(n);
        dxp = dxp.dot(np) * np / np.dot(np);

        // niter2
//        lambda = lambda * 2 * d /(n.dot(n) + np.dot(np));
//        dx = lambda * n;
//        dxp = lambda * np;

        x -= St * dx;
        xp -= St * dxp;

       // std::cout << "after " << (xp.t() * E * x) << " " << (x.t() * E * xp) << " x " << x << " xp " << xp << "\n";
//        continue;

        (*new_pts++) = x[0];
        (*new_pts++) = x[1];
        (*new_pts++) = xp[0];
        (*new_pts++) = xp[1];
//        std::cout << points.row(pt) << " vs " << new_points.row(pt) << "\n";

        if (calibrated) {
            const Vec3d z1 = x.cross(R1 * xp);
            const Vec3d z2 = x.cross(R2 * xp);
            const Vec3d X1 = (Vec3d(Mat(z1.t() * E))).dot(xp) * x / (z1.dot(z1));
            const Vec3d X2 = (Vec3d(Mat(z2.t() * E))).dot(xp) * x / (z2.dot(z2));
            (*pts3D_R1++) = X1[0];
            (*pts3D_R1++) = X1[1];
            (*pts3D_R1++) = X1[2];
            (*pts3D_R2++) = X2[0];
            (*pts3D_R2++) = X2[1];
            (*pts3D_R2++) = X2[2];
            if (X1[2] > 0) {
                num_pts_positive_depth[0]++;
                // pts3D_R1[3*pt  ] = X1[0];
                // pts3D_R1[3*pt+1] = X1[1];
                // pts3D_R1[3*pt+2] = X1[2];
            }
            if (X2[2] > 0) {
                num_pts_positive_depth[1]++;
                // pts3D_R2[3*pt  ] = X2[0];
                // pts3D_R2[3*pt+1] = X2[1];
                // pts3D_R2[3*pt+2] = X2[2];
            }
            // std::cout << X1 << " " << X2 << "\n";
        }
    }
    if (num_pts_positive_depth[0] > num_pts_positive_depth[1]) {
        R = Mat(R1);
        points3D_R1.copyTo(points3D);
    } else {
        R = Mat(R2);
        points3D_R2.copyTo(points3D);
    }
}

//////////////////////////////////////// RANDOM GENERATOR /////////////////////////////
class UniformRandomGeneratorImpl : public UniformRandomGenerator {
private:
    int subset_size = 0, max_range = 0;
    std::vector<int> subset;
    RNG rng;
public:
    explicit UniformRandomGeneratorImpl (int state) : rng(state) {}

    // interval is <0; max_range);
    UniformRandomGeneratorImpl (int state, int max_range_, int subset_size_) : rng(state) {
        subset_size = subset_size_;
        max_range = max_range_;
        subset = std::vector<int>(subset_size_);
    }

    int getRandomNumber () override {
        return rng.uniform(0, max_range);
    }

    int getRandomNumber (int max_rng) override {
        return rng.uniform(0, max_rng);
    }

    // closed range
    void resetGenerator (int max_range_) override {
        CV_CheckGE(0, max_range_, "max range must be greater than 0");
        max_range = max_range_;
    }

    void generateUniqueRandomSet (std::vector<int>& sample) override {
        CV_CheckLE(subset_size, max_range, "RandomGenerator. Subset size must be LE than range!");
        int j, num;
        sample[0] = rng.uniform(0, max_range);
        for (int i = 1; i < subset_size;) {
            num = rng.uniform(0, max_range);
            // check if value is in array
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    // if so, generate again
                    break;
            // success, value is not in array, so it is unique, add to sample.
            if (j == -1) sample[i++] = num;
        }
    }

    // interval is <0; max_range)
    void generateUniqueRandomSet (std::vector<int>& sample, int max_range_) override {
        /*
         * necessary condition:
         * if subset size is bigger than range then array cannot be unique,
         * so function has infinite loop.
         */
        CV_CheckLE(subset_size, max_range_, "RandomGenerator. Subset size must be LE than range!");
        int num, j;
        sample[0] = rng.uniform(0, max_range_);
        for (int i = 1; i < subset_size;) {
            num = rng.uniform(0, max_range_);
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    break;
            if (j == -1) sample[i++] = num;
        }
    }

    // interval is <0, max_range)
    void generateUniqueRandomSet (std::vector<int>& sample, int subset_size_, int max_range_) override {
        CV_CheckLE(subset_size_, max_range_, "RandomGenerator. Subset size must be LE than range!");
        int num, j;
        sample[0] = rng.uniform(0, max_range_);
        for (int i = 1; i < subset_size_;) {
            num = rng.uniform(0, max_range_);
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    break;
            if (j == -1) sample[i++] = num;
        }
    }
    const std::vector<int> &generateUniqueRandomSubset (std::vector<int> &array1, int size1) override {
        CV_CheckLE(subset_size, size1, "RandomGenerator. Subset size must be LE than range!");
        int temp_size1 = size1;
        for (int i = 0; i < subset_size; i++) {
            const int idx1 = rng.uniform(0, temp_size1);
            subset[i] = array1[idx1];
            std::swap(array1[idx1], array1[--temp_size1]);
        }
        return subset;
    }

    void setSubsetSize (int subset_size_) override {
        subset_size = subset_size_;
    }
    int getSubsetSize () const override { return subset_size; }
};

Ptr<UniformRandomGenerator> UniformRandomGenerator::create (int state) {
    return makePtr<UniformRandomGeneratorImpl>(state);
}
Ptr<UniformRandomGenerator> UniformRandomGenerator::create
        (int state, int max_range, int subset_size_) {
    return makePtr<UniformRandomGeneratorImpl>(state, max_range, subset_size_);
}

// @k_minth - desired k-th minimal element. For median is half of array
// closed working interval of array <@left; @right>
float quicksort_median (std::vector<float> &array, int k_minth, int left, int right);
float quicksort_median (std::vector<float> &array, int k_minth, int left, int right) {
    // length is 0, return single value
    if (right - left == 0) return array[left];

    // get pivot, the rightest value in array
    const auto pivot = array[right];
    int right_ = right - 1; // -1, not including pivot
    // counter of values smaller equal than pivot
    int j = left, values_less_eq_pivot = 1; // 1, inludes pivot already
    for (; j <= right_;) {
        if (array[j] <= pivot) {
            j++;
            values_less_eq_pivot++;
        } else
            // value is bigger than pivot, swap with right_ value
            // swap values in array and decrease interval
            std::swap(array[j], array[right_--]);
    }
    if (values_less_eq_pivot == k_minth) return pivot;
    if (k_minth > values_less_eq_pivot)
        return quicksort_median(array, k_minth - values_less_eq_pivot, j, right-1);
    else
        return quicksort_median(array, k_minth, left, j-1);
}

// find median using quicksort with complexity O(log n)
// Note, function changes order of values in array
float Utils::findMedian (std::vector<float> &array) {
    const int length = static_cast<int>(array.size());
    if (length % 2) {
        // odd number of values
        return quicksort_median (array, length/2+1, 0, length-1);
    } else {
        // even: return average
        return (quicksort_median(array, length/2  , 0, length-1) +
                quicksort_median(array, length/2+1, 0, length-1))/2;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Radius Search Graph /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class RadiusSearchNeighborhoodGraphImpl : public RadiusSearchNeighborhoodGraph {
private:
    std::vector<std::vector<int>> graph;
public:
    RadiusSearchNeighborhoodGraphImpl (const Mat &container_, int points_size,
                               double radius, int flann_search_params, int num_kd_trees) {
        // Radius search OpenCV works only with float data
        CV_Assert(container_.type() == CV_32F);

        FlannBasedMatcher flann(makePtr<flann::KDTreeIndexParams>(num_kd_trees), makePtr<flann::SearchParams>(flann_search_params));
        std::vector<std::vector<DMatch>> neighbours;
        flann.radiusMatch(container_, container_, neighbours, (float)radius);

        // allocate graph
        graph = std::vector<std::vector<int>> (points_size);

        int pt = 0;
        for (const auto &n : neighbours) {
            if (n.size() <= 1)
                continue;
            auto &graph_row = graph[pt];
            graph_row = std::vector<int>(n.size()-1);
            int j = 0;
            for (const auto &idx : n)
                // skip neighbor which has the same index as requested point
                if (idx.trainIdx != pt)
                    graph_row[j++] = idx.trainIdx;
            pt++;
        }
    }

    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        return graph[point_idx];
    }
};
Ptr<RadiusSearchNeighborhoodGraph> RadiusSearchNeighborhoodGraph::create (const Mat &points,
        int points_size, double radius_, int flann_search_params, int num_kd_trees) {
    return makePtr<RadiusSearchNeighborhoodGraphImpl> (points, points_size, radius_,
            flann_search_params, num_kd_trees);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// FLANN Graph /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class FlannNeighborhoodGraphImpl : public FlannNeighborhoodGraph {
private:
    std::vector<std::vector<int>> graph;
    std::vector<std::vector<double>> distances;
public:
    FlannNeighborhoodGraphImpl (const Mat &container_, int points_size, int k_nearest_neighbors,
            bool get_distances, int flann_search_params_, int num_kd_trees) {
        CV_Assert(k_nearest_neighbors <= points_size);
        // FLANN works only with float data
        CV_Assert(container_.type() == CV_32F);

        flann::Index flannIndex (container_.reshape(1), flann::KDTreeIndexParams(num_kd_trees));
        Mat dists, nearest_neighbors;

        flannIndex.knnSearch(container_, nearest_neighbors, dists, k_nearest_neighbors+1,
                flann::SearchParams(flann_search_params_));

        // first nearest neighbor of point is this point itself.
        // remove this first column
        nearest_neighbors.colRange(1, k_nearest_neighbors+1).copyTo (nearest_neighbors);

        graph = std::vector<std::vector<int>>(points_size, std::vector<int>(k_nearest_neighbors));
        const auto * const nn = (int *) nearest_neighbors.data;
        const auto * const dists_ptr = (float *) dists.data;

        if (get_distances)
            distances = std::vector<std::vector<double>>(points_size, std::vector<double>(k_nearest_neighbors));

        for (int pt = 0; pt < points_size; pt++) {
            std::copy(nn + k_nearest_neighbors*pt, nn + k_nearest_neighbors*pt + k_nearest_neighbors, &graph[pt][0]);
            if (get_distances)
                std::copy(dists_ptr + k_nearest_neighbors*pt, dists_ptr + k_nearest_neighbors*pt + k_nearest_neighbors,
                          &distances[pt][0]);
        }
    }
    const std::vector<double>& getNeighborsDistances (int idx) const override {
        return distances[idx];
    }
    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        // CV_Assert(point_idx_ < num_vertices);
        return graph[point_idx];
    }
};

Ptr<FlannNeighborhoodGraph> FlannNeighborhoodGraph::create(const Mat &points,
           int points_size, int k_nearest_neighbors_, bool get_distances,
           int flann_search_params_, int num_kd_trees) {
    return makePtr<FlannNeighborhoodGraphImpl>(points, points_size,
        k_nearest_neighbors_, get_distances, flann_search_params_, num_kd_trees);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Grid Neighborhood Graph /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class GridNeighborhoodGraphImpl : public GridNeighborhoodGraph {
private:
    // This struct is used for the nearest neighbors search by griding two images.
    struct CellCoord {
        int c1x, c1y, c2x, c2y;
        CellCoord (int c1x_, int c1y_, int c2x_, int c2y_) {
            c1x = c1x_; c1y = c1y_; c2x = c2x_; c2y = c2y_;
        }
        bool operator==(const CellCoord &o) const {
            return c1x == o.c1x && c1y == o.c1y && c2x == o.c2x && c2y == o.c2y;
        }
        bool operator<(const CellCoord &o) const {
            if (c1x < o.c1x) return true;
            if (c1x == o.c1x && c1y < o.c1y) return true;
            if (c1x == o.c1x && c1y == o.c1y && c2x < o.c2x) return true;
            return c1x == o.c1x && c1y == o.c1y && c2x == o.c2x && c2y < o.c2y;
        }
    };

    std::map<CellCoord, std::vector<int >> neighbors_map;
    std::vector<std::vector<int>> graph;
public:
    GridNeighborhoodGraphImpl (const Mat &container_, int points_size,
          int cell_size_x_img1, int cell_size_y_img1, int cell_size_x_img2, int cell_size_y_img2,
          int max_neighbors) {

        const auto * const container = (float *) container_.data;
        // <int, int, int, int> -> {neighbors set}
        // Key is cell position. The value is indexes of neighbors.

        const float cell_sz_x1 = 1.f / (float) cell_size_x_img1,
                    cell_sz_y1 = 1.f / (float) cell_size_y_img1,
                    cell_sz_x2 = 1.f / (float) cell_size_x_img2,
                    cell_sz_y2 = 1.f / (float) cell_size_y_img2;
        const int dimension = container_.cols;
        for (int i = 0; i < points_size; i++) {
            const int idx = dimension * i;
            neighbors_map[CellCoord((int)(container[idx  ] * cell_sz_x1),
                                    (int)(container[idx+1] * cell_sz_y1),
                                    (int)(container[idx+2] * cell_sz_x2),
                                    (int)(container[idx+3] * cell_sz_y2))].emplace_back(i);
        }

        //--------- create a graph ----------
        graph = std::vector<std::vector<int>>(points_size);

        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map) {
            const int neighbors_in_cell = static_cast<int>(cell.second.size());
            // only one point in cell -> no neighbors
            if (neighbors_in_cell < 2) continue;

            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
            for (int v_in_cell : neighbors) {
                // there is always at least one neighbor
                auto &graph_row = graph[v_in_cell];
                graph_row = std::vector<int>(std::min(max_neighbors, neighbors_in_cell-1));
                int j = 0;
                for (int n : neighbors)
                    if (n != v_in_cell){
                        graph_row[j++] = n;
                        if (j >= max_neighbors)
                            break;
                    }
            }
        }
    }
    const std::vector<std::vector<int>> &getGraph () const override { return graph; }
    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        // Note, neighbors vector also includes point_idx!
        // return neighbors_map[vertices_to_cells[point_idx]];
        return graph[point_idx];
    }
};

Ptr<GridNeighborhoodGraph> GridNeighborhoodGraph::create(const Mat &points,
     int points_size, int cell_size_x_img1_, int cell_size_y_img1_,
     int cell_size_x_img2_, int cell_size_y_img2_, int max_neighbors) {
    return makePtr<GridNeighborhoodGraphImpl>(points, points_size,
      cell_size_x_img1_, cell_size_y_img1_, cell_size_x_img2_, cell_size_y_img2_, max_neighbors);
}

class GridNeighborhoodGraph2Impl : public GridNeighborhoodGraph2 {
private:
    // This struct is used for the nearest neighbors search by griding two images.
    struct CellCoord {
        int c1x, c1y;
        CellCoord (int c1x_, int c1y_) {
            c1x = c1x_; c1y = c1y_;
        }
        bool operator==(const CellCoord &o) const {
            return c1x == o.c1x && c1y == o.c1y;
        }
        bool operator<(const CellCoord &o) const {
            if (c1x < o.c1x) return true;
            return c1x == o.c1x && c1y < o.c1y;
        }
    };

    std::map<CellCoord, std::vector<int >> neighbors_map1, neighbors_map2;
    std::vector<std::vector<int>> graph;
public:
    GridNeighborhoodGraph2Impl (const Mat &container_, int points_size,
            int cell_size_x_img1, int cell_size_y_img1, int cell_size_x_img2, int cell_size_y_img2) {

        const auto * const container = (float *) container_.data;
        // <int, int, int, int> -> {neighbors set}
        // Key is cell position. The value is indexes of neighbors.

        const float cell_sz_x1 = 1.f / (float) cell_size_x_img1,
                cell_sz_y1 = 1.f / (float) cell_size_y_img1,
                cell_sz_x2 = 1.f / (float) cell_size_x_img2,
                cell_sz_y2 = 1.f / (float) cell_size_y_img2;
        const int dimension = container_.cols;
        for (int i = 0; i < points_size; i++) {
            const int idx = dimension * i;
            neighbors_map1[CellCoord((int)(container[idx  ] * cell_sz_x1),
                                    (int)(container[idx+1] * cell_sz_y1))].emplace_back(i);
            neighbors_map2[CellCoord((int)(container[idx+2] * cell_sz_x2),
                                    (int)(container[idx+3] * cell_sz_y2))].emplace_back(i);
        }

        //--------- create a graph ----------
        graph = std::vector<std::vector<int>>(points_size);

//        std::cout << points_size << " " <<  cell_size_x_img1 << " cell size \n";
        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map1) {
            const int neighbors_in_cell = static_cast<int>(cell.second.size());
//            std::cout << "1) neighbors_in_cell " << neighbors_in_cell << "\n";
            // only one point in cell -> no neighbors
            if (neighbors_in_cell < 2) continue;

//            std::cout << neighbors_in_cell << " neighbors\n";
            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
            for (int v_in_cell : neighbors) {
                // there is always at least one neighbor
                auto &graph_row = graph[v_in_cell];
                graph_row = std::vector<int>(neighbors_in_cell);
                int j = 0;
                for (int n : neighbors)
                    if (n != v_in_cell){
                        graph_row[j++] = n;
                    }
            }
        }

        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map2) {
//            std::cout << "2) neighbors_in_cell " << cell.second.size() << "\n";
            if (cell.second.size() < 2) continue;
            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
            for (int v_in_cell : neighbors) {
                // there is always at least one neighbor
                auto &graph_row = graph[v_in_cell];
                for (int n : neighbors)
                    if (n != v_in_cell){
                        graph_row.emplace_back(n);
                    }
            }
        }
    }
    const std::vector<std::vector<int>> &getGraph () const override { return graph; }
    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        // Note, neighbors vector also includes point_idx!
        // return neighbors_map[vertices_to_cells[point_idx]];
        return graph[point_idx];
    }
};

Ptr<GridNeighborhoodGraph2> GridNeighborhoodGraph2::create(const Mat &points,
        int points_size, int cell_size_x_img1_, int cell_size_y_img1_, int cell_size_x_img2_, int cell_size_y_img2_) {
    return makePtr<GridNeighborhoodGraph2Impl>(points, points_size,
            cell_size_x_img1_, cell_size_y_img1_, cell_size_x_img2_, cell_size_y_img2_);
}
}}

namespace vsac {
bool getCorrectedPointsHomography(const cv::Mat &points, cv::Mat &corr_points, const cv::Mat &H) {
    std::vector<bool> mask(points.rows, true);
    return getCorrectedPointsHomography(points, corr_points, H, mask);
}

bool getCorrectedPointsHomography(const cv::Mat &points, cv::Mat &corr_points, const cv::Mat &H, const std::vector<bool> &mask) {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> h((double *)H.data);
    h /= h(2,2); // must be normaized to avoid internal sqrt error
    // std::cout << h << '\n';
    Eigen::Matrix<double, 3, 3> sqrtH;
    try {
        sqrtH = h.sqrt();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        std::cerr << "Cannot find matrix square root!\n";
        return false;
    }
    cv::Mat_<double > H_half (3,3,(double*)sqrtH.data());
    cv::transpose(H_half, H_half);
//    const auto * const pts = (float *) points.data;

    cv::Mat pts1, pts2, mid_pts, corr_pts1, corr_pts2;
    cv::vconcat(points.colRange(0,2).t(), cv::Mat_<float>::ones(1, points.rows), pts1);
    cv::vconcat(points.colRange(2,4).t(), cv::Mat_<float>::ones(1, points.rows), pts2);
    pts1.convertTo(pts1, CV_64F);
    pts2.convertTo(pts2, CV_64F);
    corr_pts1 = H_half       * pts1;
    corr_pts2 = H_half.inv() * pts2;
    cv::divide(corr_pts1.row(0), corr_pts1.row(2), corr_pts1.row(0));
    cv::divide(corr_pts1.row(1), corr_pts1.row(2), corr_pts1.row(1));
    cv::divide(corr_pts2.row(0), corr_pts2.row(2), corr_pts2.row(0));
    cv::divide(corr_pts2.row(1), corr_pts2.row(2), corr_pts2.row(1));
    // mid_pts = (H_half * pts1 + H_half.inv() * pts2) * 0.5;
    mid_pts = (corr_pts1 + corr_pts2) * 0.5;
    corr_pts1 = H_half.inv() * mid_pts;
    corr_pts2 = H_half       * mid_pts;
    cv::divide(corr_pts1.row(0), corr_pts1.row(2), corr_pts1.row(0));
    cv::divide(corr_pts1.row(1), corr_pts1.row(2), corr_pts1.row(1));
    cv::divide(corr_pts2.row(0), corr_pts2.row(2), corr_pts2.row(0));
    cv::divide(corr_pts2.row(1), corr_pts2.row(2), corr_pts2.row(1));
    cv::hconcat(corr_pts1.rowRange(0,2).t(), corr_pts2.rowRange(0,2).t(), corr_points);
    corr_points.convertTo(corr_points, CV_32F);
    return true;
}
}