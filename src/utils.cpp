#include "precomp.hpp"
#include <opencv2/flann/miniflann.hpp>

namespace cv { namespace vsac {
int mergePoints (InputArray pts1_, InputArray pts2_, Mat &pts, bool ispnp) {
    Mat pts1 = pts1_.getMat(), pts2 = pts2_.getMat();
    auto convertPoints = [] (Mat &points, int pt_dim) {
        points.convertTo(points, CV_32F); // convert points to have float precision
        if (points.channels() > 1)
            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
        if (points.rows < points.cols)
            transpose(points, points); // transpose so points will be in rows
        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
            points = points.colRange(0, pt_dim);
    };

    convertPoints(pts1, 2); // pts1 are always image points
    convertPoints(pts2, ispnp ? 3 : 2); // for PnP points are 3D

    // points are of size [Nx2 Nx2] = Nx4 for H, F, E
    // points are of size [Nx2 Nx3] = Nx5 for PnP
    hconcat(pts1, pts2, pts);
    return pts.rows;
}

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

// since F(E) has rank 2 we use cross product to compute epipole,
// since the third column / row is linearly dependent on two first
// this is faster than SVD
Vec3d Utils::getLeftEpipole (const Mat &F/*E*/) {
    Vec3d _e = F.col(0).cross(F.col(2)); // F^T e' = 0; e'^T F = 0
    const auto * const e = _e.val;
    if (e[0] <= DBL_EPSILON && e[0] > -DBL_EPSILON &&
        e[1] <= DBL_EPSILON && e[1] > -DBL_EPSILON &&
        e[2] <= DBL_EPSILON && e[2] > -DBL_EPSILON)
        _e = Vec3d(Mat(F.col(1))).cross(F.col(2));  // if e' is zero
    return _e; // e'
}
Vec3d Utils::getRightEpipole (const Mat &F/*E*/) {
    Vec3d _e = F.row(0).cross(F.row(2)); // Fe = 0
    const auto * const e = _e.val;
    if (e[0] <= DBL_EPSILON && e[0] > -DBL_EPSILON &&
        e[1] <= DBL_EPSILON && e[1] > -DBL_EPSILON &&
        e[2] <= DBL_EPSILON && e[2] > -DBL_EPSILON)
        _e = F.row(1).cross(F.row(2));  // if e is zero
    return _e;
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
     return {0,    -v[2], v[1],
             v[2],  0,   -v[0],
            -v[1],  v[0], 0};
}

Matx33d Math::rotVec2RotMat (const Vec3d &v) {
    const double phi = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    const double x = v[0] / phi, y = v[1] / phi, z = v[2] / phi;
    const double a = sin(phi), b = cos(phi);
    // R = I + sin(phi) * skew(v) + (1 - cos(phi) * skew(v)^2
    return {(b - 1)*y*y + (b - 1)*z*z + 1, -a*z - x*y*(b - 1), a*y - x*z*(b - 1),
     a*z - x*y*(b - 1), (b - 1)*x*x + (b - 1)*z*z + 1, -a*x - y*z*(b - 1),
    -a*y - x*z*(b - 1), a*x - y*z*(b - 1), (b - 1)*x*x + (b - 1)*y*y + 1};
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
        if (fabs(pivot) < DBL_EPSILON)
            continue;

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
            
        // since P1 = K [I | 0] it imples that if scale1 < 0 then z < 0
        if (scale1 > 0 && scale2 > 0) {
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

    R = correct_idx <= 1 ? Mat(R1) : Mat(R2);
    // 0 and 2 index corresponds to +t, 1 and 3 to -t
    t_vec = correct_idx % 2 ? Mat(-t) : Mat(t);
    pts3D[correct_idx].copyTo(points3D);
    good_mask = good_masks[correct_idx];
    depths1 = depths1_[correct_idx];
    depths2 = depths2_[correct_idx];
}

// https://www.researchgate.net/publication/221362655_Triangulation_Made_Easy
void Utils::triangulatePoints (const Mat &F_, const Mat &points1, const Mat &points2, Mat &corr_points1, Mat &corr_points2,
        const Mat &K1_, const Mat &K2_, Mat &points3D, Mat &R, Mat &t_out, const std::vector<bool> &good_point_mask) {
    cv::Mat pts1_ = points1, pts2_ = points2;
    pts1_.convertTo(pts1_, CV_32F);
    pts2_.convertTo(pts2_, CV_32F);
    corr_points1 = Mat_<float>::zeros(points1.rows, 2);
    corr_points2 = Mat_<float>::zeros(points1.rows, 2);

    int correct_pose_pair = -1, num_pts_processed = 0;
    int num_pts_positive_depth[4] = {0};
    cv::Mat points3D_R1, points3D_R2;
    Vec3d t_, t_corr;
    Matx33d R1, R2, F (F_), K1, K2, R_corr, K1_inv, K2_inv, E;
    const bool has_calibration = !K1_.empty() && !K2_.empty();
    if (has_calibration) {
        K1 = Matx33d(K1_);
        K2 = Matx33d(K2_);
        K2_inv = K2.inv();
        K1_inv = K1.inv();
        E = K2.t() * F * K1;
        points3D_R1 = Mat_<float>::zeros(points1.rows, 3);
        points3D_R2 = Mat_<float>::zeros(points1.rows, 3);
        decomposeEssentialMat(E, R1, R2, t_);
    }
    F = F.t();
    auto * pts3D_R1 = (float *) points3D_R1.data, * pts3D_R2 = (float *) points3D_R2.data;
    const auto * const pts1 = (float *) pts1_.data, * const pts2 = (float *) pts2_.data;
    auto * cpts1 = (float *) corr_points1.data, * cpts2 = (float *) corr_points2.data;

    const Matx23d S (1, 0, 0,
                     0, 1, 0);
    const Matx32d St = S.t();
    const Matx23d SF = S * F, SFt = S * F.t();
    const Matx22d SFSt = S * F * St;
    for (int pt = 0; pt < points1.rows; pt++) {
        if (!good_point_mask[pt]) continue;
        const int idx = 2*pt;
        Vec3d x  (pts1[idx], pts1[idx+1], 1);
        Vec3d xp (pts2[idx], pts2[idx+1], 1);
        Vec2d n = SF * xp, np = SFt * x;
        // n^T SES^T n'
        const auto a = (n[0] * SFSt(0,0) + n[1] * SFSt(1,0)) * np[0] +
                       (n[0] * SFSt(0,1) + n[1] * SFSt(1,1)) * np[1];
        const auto b = 0.5 * (n.dot(n) + np.dot(np));
        const auto c = (x[0] * F(0,0) + x[1] * F(1,0) + F(2,0)) * xp[0] +
                       (x[0] * F(0,1) + x[1] * F(1,1) + F(2,1)) * xp[1] +
                       (x[0] * F(0,2) + x[1] * F(1,2) + F(2,2)); // xT E x
        auto lambda = c / (b + sqrt(b*b - a * c));
        auto dx = lambda * n;
        auto dxp = lambda * np;
        n -= SFSt * dxp;
        np -= SFSt.t() * dx;

        // niter1
        dx = dx.dot(n) * n / n.dot(n);
        dxp = dxp.dot(np) * np / np.dot(np);

        // niter2
        // lambda = lambda * 2 * d /(n.dot(n) + np.dot(np));
        // dx = lambda * n;
        // dxp = lambda * np;

        x -= St * dx;
        xp -= St * dxp;

        cpts1[idx  ] = x[0];
        cpts1[idx+1] = x[1];
        cpts2[idx  ] = xp[0];
        cpts2[idx+1] = xp[1];

        if (has_calibration) {
            num_pts_processed++;
            x[2] = 1; xp[2] = 1;
            // get normalized points by K^-1 x
            cv::Vec3d norm_x1 = K1_inv * x, norm_x2 = K2_inv * xp;
            cv::Vec3d norm_x1_unit = norm_x1 / norm(norm_x1), norm_x2_unit = norm_x2 / norm(norm_x2);
            if (correct_pose_pair != -1) {
                // we found correct pose
                const Vec3d z = norm_x1.cross(R_corr * norm_x2);
                const Vec3d X = z.dot(E * norm_x2) * x / (z.dot(z));
                pts3D_R1[3*pt  ] = X[0];
                pts3D_R1[3*pt+1] = X[1];
                pts3D_R1[3*pt+2] = X[2];
            } else {
                const bool R1_t_good     = Utils::satisfyCheirality(R1, t_, norm_x1_unit, norm_x2_unit);
                const bool R1_min_t_good = Utils::satisfyCheirality(R1,-t_, norm_x1_unit, norm_x2_unit);
                const bool R2_t_good     = Utils::satisfyCheirality(R2, t_, norm_x1_unit, norm_x2_unit);
                const bool R2_min_t_good = Utils::satisfyCheirality(R2,-t_, norm_x1_unit, norm_x2_unit);
                if (R1_t_good)     num_pts_positive_depth[0]++;
                if (R1_min_t_good) num_pts_positive_depth[1]++;
                if (R2_t_good)     num_pts_positive_depth[2]++;
                if (R2_min_t_good) num_pts_positive_depth[3]++;
                if (R1_t_good || R1_min_t_good) {
                    const Vec3d z = norm_x1.cross(R1 * norm_x2);
                    const Vec3d X = z.dot(E * norm_x2) * x / (z.dot(z));
                    pts3D_R1[3*pt  ] = X[0];
                    pts3D_R1[3*pt+1] = X[1];
                    pts3D_R1[3*pt+2] = X[2];
                } else {
                    const Vec3d z = norm_x1.cross(R2 * norm_x2);
                    const Vec3d X = z.dot(E * norm_x2) * x / (z.dot(z));
                    pts3D_R2[3*pt  ] = X[0];
                    pts3D_R2[3*pt+1] = X[1];
                    pts3D_R2[3*pt+2] = X[2];
                }
            }
            if (num_pts_processed == 100 || num_pts_processed == points1.rows) {
                // 100 points is enough to find out good pose pair
                int max_points_in_front = 0;
                for (int cam = 0; cam < 4; cam++) {
                    if (max_points_in_front < num_pts_positive_depth[cam]) {
                        max_points_in_front = num_pts_positive_depth[cam];
                        correct_pose_pair = cam;
                    }
                }
                t_corr = correct_pose_pair % 2 ? -t_ : t_;
                if (correct_pose_pair >= 2) {
                    R_corr = R2;
                    // we store correct points in points3D_R1 array, so
                    // copy all elements of points3D_R2.
                    std::copy((float *)points3D_R2.data, (float *)points3D_R2.data+3*pt, (float *)points3D_R1.data);
                } else R_corr = R1;
            }
        }
    }
    R = Mat(R_corr);
    t_out = Mat(t_corr);
    points3D_R1.copyTo(points3D);
    points3D.convertTo(points3D, points1.type());
    corr_points1.convertTo(corr_points1, points1.type());
    corr_points2.convertTo(corr_points2, points2.type());
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
        if (subset_size < subset_size_)
            subset.resize(subset_size_);
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
                quicksort_median(array, length/2+1, 0, length-1))*.5f;
    }
}

///////////////////////////////// Radius Search Graph /////////////////////////////////////////////
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

///////////////////////////////// FLANN Graph /////////////////////////////////////////////
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

///////////////////////////////// Grid Neighborhood Graph /////////////////////////////////////////
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

        const auto cell_sz_x1 = 1.f / (float) cell_size_x_img1,
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

        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map1) {
            const int neighbors_in_cell = static_cast<int>(cell.second.size());
            // only one point in cell -> no neighbors
            if (neighbors_in_cell < 2) continue;

            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
//            for (int v_in_cell : neighbors) {
            const int v_in_cell = neighbors[0];
                // there is always at least one neighbor
                auto &graph_row = graph[v_in_cell];
                graph_row.reserve(neighbors_in_cell);
                for (int n : neighbors)
                    if (n != v_in_cell)
                        graph_row.emplace_back(n);
//            }
        }

        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map2) {
            if (cell.second.size() < 2) continue;
            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
//            for (const int &v_in_cell : neighbors) {
            const int v_in_cell = neighbors[0];
                // there is always at least one neighbor
                auto &graph_row = graph[v_in_cell];
                for (const int &n : neighbors)
                    if (n != v_in_cell) {
                        bool has = false;
                        for (const int &nn : graph_row)
                            if (n == nn) {
                                has = true; break;
                            }
                        if (!has) graph_row.emplace_back(n);
                    }
//            }
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
void triangulatePointsLindstrom (const cv::Mat &F, const cv::Mat &points1, const cv::Mat &points2, cv::Mat &points1_corr,
        cv::Mat &points2_corr, const std::vector<bool> &good_point_mask) {
    cv::Mat temp;
    cv::vsac::Utils::triangulatePoints(F, points1, points2, points1_corr, points2_corr, temp, temp, temp, temp, temp, good_point_mask);
}
void triangulatePointsLindstrom (const cv::Mat &E, const cv::Mat &points1, const cv::Mat &points2, cv::Mat &points1_corr,
        cv::Mat &points2_corr, const cv::Mat &K1, const cv::Mat &K2, cv::Mat &points3D, cv::Mat &R, cv::Mat &t, const std::vector<bool> &good_point_mask) {
    cv::vsac::Utils::triangulatePoints(E, points1, points2, points1_corr, points2_corr, K1, K2, points3D, R, t, good_point_mask);
}

bool getCorrectedPointsHomography(const cv::Mat &points1, const cv::Mat &points2, cv::Mat &corr_points1, cv::Mat &corr_points2, const cv::Mat &H, const std::vector<bool> &mask) {
    cv::Mat pts1 = points1, pts2 = points2;
    pts1.convertTo(pts1, CV_32F);
    pts2.convertTo(pts2, CV_32F);
    const auto * const p1 = (float *) pts1.data;
    const auto * const p2 = (float *) pts2.data;
    corr_points1 = cv::Mat_<float>::zeros(points1.rows, 2);
    corr_points2 = cv::Mat_<float>::zeros(points2.rows, 2);
    auto * cp1 = (float *) corr_points1.data;
    auto * cp2 = (float *) corr_points2.data;
    cv::Mat H_inv = H.inv();
    const auto * const h = (double *) H.data;
    const auto * const h_inv = (double *) H_inv.data;
    // https://cmp.felk.cvut.cz/~chum/papers/chum-icpr12.pdf
    double s2_sqr = 1, EPS = 1;
    for (int i = 0; i < points1.rows; i++) {
        if (! mask[i]) continue;
        const int idx = 2*i;
        const auto x1 = p1[idx], y1 = p1[idx+1], x2 = p2[idx], y2 = p2[idx+1];
        // Jacobian of homography matrix
        cv::Matx33d A (h[0] - x2*h[6], h[1] - x2*h[7], h[2] + x2 * (x1 * h[6] + y1 * h[7]),
                       h[3] - y2*h[6], h[4] - y2*h[7], h[5] + y2 * (x1 * h[6] + y1 * h[7]),
                       0, 0, h[8] + h[6] * x1 + h[7] * y1);
        const double s1_sqr = pow(1/cv::determinant(A / A(2,2)), 2);
        // H^-1 (x2 y2 1)
        const auto z1_proj =  h_inv[6] * x2 + h_inv[7] * y2 + h_inv[8];
        const auto x1_proj = (h_inv[0] * x2 + h_inv[1] * y2 + h_inv[2]) / z1_proj;
        const auto y1_proj = (h_inv[3] * x2 + h_inv[4] * y2 + h_inv[5]) / z1_proj;

        const auto x1_corr = (s1_sqr * x1 + s2_sqr * x1_proj) / (s1_sqr + s2_sqr);
        const auto y1_corr = (s1_sqr * y1 + s2_sqr * y1_proj) / (s1_sqr + s2_sqr);

        const auto z2_corr =  h[6] * x1_corr + h[7] * y1_corr + h[8];
        const auto x2_corr = (h[0] * x1_corr + h[1] * y1_corr + h[2]) / z2_corr;
        const auto y2_corr = (h[3] * x1_corr + h[4] * y1_corr + h[5]) / z2_corr;

        cp1[idx  ] = x1_corr;
        cp1[idx+1] = y1_corr;
        cp2[idx  ] = x2_corr;
        cp2[idx+1] = y2_corr;
    }

    corr_points1.convertTo(corr_points1, points1.type());
    corr_points2.convertTo(corr_points2, points2.type());
    return true;
}
/*
 //
 // Older version for correcting points using half homography
 //
bool getCorrectedPointsHomography(const cv::Mat &points, cv::Mat &corr_points, const cv::Mat &H, const std::vector<bool> &mask) {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> h((double *)H.data);
    h /= h(2,2); // must be normaized to avoid internal sqrt error
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
}*/

}