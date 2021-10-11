#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "../include/vsac_definitions.hpp"

int convertPoints (const pybind11::array_t<double> &pts1, const pybind11::array_t<double> &pts2,
                    cv::Mat &points1, cv::Mat &points2) {
    assert(pts1.size() != 1 && pts2.size() != 1);
    const int points_size = pts1.request().shape[0];
    const int cols1 = pts1.request().shape[1], cols2 = pts2.request().shape[1];
    const int num_cols = (int) (pts1.request().strides[0] / pts1.request().strides[1]);
    if (num_cols == 4 || num_cols == 5) {
        cv::Mat points (points_size, num_cols, CV_64F, (double*)pts1.request().ptr);
        points.colRange(0,2).copyTo(points1);
        points.colRange(2, num_cols).copyTo(points2);
    } else if (num_cols == 2) {
        points1 = cv::Mat(points_size, cols1, CV_64F, (double*)pts1.request().ptr);
        points2 = cv::Mat(points_size, cols2, CV_64F, (double*)pts2.request().ptr);
    } else {
        std::cerr << "Incorrect size of points! Must be [N x 2]\n";
        exit(1);
    }
    return points_size;
}

pybind11::tuple estimate(const vsac::Params &params,
        const pybind11::array_t<double> &pts1, const pybind11::array_t<double> &pts2,
        pybind11::array_t<double> &K1, pybind11::array_t<double> &K2,
        pybind11::array_t<double> &d_coef1, pybind11::array_t<double> &d_coef2) {
    cv::Mat points1, points2;
    const int points_size = convertPoints(pts1, pts2, points1, points2);

    cv::Mat distortion_coeff1, distortion_coeff2, _K1, _K2;
    if (K1.size() == 9) _K1 = cv::Mat_<double>(3,3, (double *)K1.request().ptr);
    if (K2.size() == 9) _K2 = cv::Mat_<double>(3,3, (double *)K2.request().ptr);
    if (d_coef1.size() > 1) distortion_coeff1 = cv::Mat_<double>(d_coef1.request().shape[0],d_coef1.request().shape[1], (double *)d_coef1.request().ptr);
    if (d_coef2.size() > 1) distortion_coeff2 = cv::Mat_<double>(d_coef2.request().shape[0],d_coef2.request().shape[1], (double *)d_coef2.request().ptr);

    pybind11::array_t<double> model_out;
    pybind11::array_t<bool> inliers_out;
    vsac::Output output;
    if (vsac::estimate(params, points1, points2, output, _K1, _K2, distortion_coeff1, distortion_coeff2)) {
        const auto &model = output.model;
        // copy model
        model_out = pybind11::array_t<double>({model.rows, model.cols});
        std::copy((double*)model.data, (double*)model.data+model.rows*model.cols, (double*)model_out.request().ptr);

        if (params.isMaskRequired()) {
            // copy inliers
            inliers_out = pybind11::array_t<bool>(points_size);
            std::copy(output.inliers_mask.begin(), output.inliers_mask.end(), (bool*)inliers_out.request().ptr);
        }
    }
    return pybind11::make_tuple(model_out, inliers_out);
}

pybind11::tuple getCorrectedPointsHomography
        (const pybind11::array_t<double> &points1, const pybind11::array_t<double> &points2,
         const pybind11::array_t<double> &H_, const pybind11::array_t<bool> &good_point_mask) {
    assert(H_.size() != 1);
    cv::Mat pts1, pts2, corr_pts1, corr_pts2, H = cv::Mat_<double>(3,3, (double *)H_.request().ptr);
    const int points_size = convertPoints(points1, points2, pts1, pts2);
    std::vector<bool> mask((bool *)good_point_mask.request().ptr, (bool *)good_point_mask.request().ptr+points_size);
    vsac::getCorrectedPointsHomography(pts1, pts2, corr_pts1, corr_pts2, H, mask);

    // copy corrected points
    pybind11::array_t<double> points1_corr = pybind11::array_t<double>({corr_pts1.rows, corr_pts1.cols});
    pybind11::array_t<double> points2_corr = pybind11::array_t<double>({corr_pts2.rows, corr_pts2.cols});
    std::copy((double*)corr_pts1.data, (double*)corr_pts1.data+corr_pts1.rows*corr_pts1.cols, (double*)points1_corr.request().ptr);
    std::copy((double*)corr_pts2.data, (double*)corr_pts2.data+corr_pts2.rows*corr_pts2.cols, (double*)points2_corr.request().ptr);
    return pybind11::make_tuple(points1_corr, points2_corr);
}

pybind11::tuple triangulatePointsLindstromWithK (const pybind11::array_t<double> &F_,
        const pybind11::array_t<double> &points1, const pybind11::array_t<double> &points2,
        const pybind11::array_t<double> &K1_, const pybind11::array_t<double> &K2_,
        const pybind11::array_t<bool> &good_point_mask) {
    const bool has_calibration = K1_.size() == 9 && K2_.size() == 9;
    assert(F_.size() != 1);
    cv::Mat pts1, pts2, corr_pts1, corr_pts2, F = cv::Mat_<double>(3,3, (double *)F_.request().ptr), K1, K2, pts3D_, R_, t_;
    if (has_calibration) {
        K1 = cv::Mat_<double>(3,3, (double *)K1_.request().ptr);
        K2 = cv::Mat_<double>(3,3, (double *)K2_.request().ptr);
    }
    const int points_size = convertPoints(points1, points2, pts1, pts2);
    std::vector<bool> mask((bool *)good_point_mask.request().ptr, (bool *)good_point_mask.request().ptr+points_size);
    pybind11::array_t<double> R, t, pts3D;
    if (has_calibration) {
        vsac::triangulatePointsLindstrom(F, pts1, pts2, corr_pts1, corr_pts2, K1, K2, pts3D_, R_, t_, mask);
        pts3D = pybind11::array_t<double>({points_size, 3});
        R = pybind11::array_t<double>({3, 3});
        t = pybind11::array_t<double>({3, 1});
        std::copy((double*)pts3D_.data, (double*)pts3D_.data+pts3D_.rows*pts3D_.cols, (double*)pts3D.request().ptr);
        std::copy((double*)R_.data, (double*)R_.data+9, (double*)R.request().ptr);
        std::copy((double*)t_.data, (double*)t_.data+3, (double*)t.request().ptr);
    } else vsac::triangulatePointsLindstrom(F, pts1, pts2, corr_pts1, corr_pts2, mask);

    // copy corrected points
    pybind11::array_t<double> points1_corr = pybind11::array_t<double>({corr_pts1.rows, corr_pts1.cols});
    pybind11::array_t<double> points2_corr = pybind11::array_t<double>({corr_pts2.rows, corr_pts2.cols});
    std::copy((double*)corr_pts1.data, (double*)corr_pts1.data+corr_pts1.rows*corr_pts1.cols, (double*)points1_corr.request().ptr);
    std::copy((double*)corr_pts2.data, (double*)corr_pts2.data+corr_pts2.rows*corr_pts2.cols, (double*)points2_corr.request().ptr);
    return has_calibration ? pybind11::make_tuple(points1_corr, points2_corr, pts3D, R, t) :
                             pybind11::make_tuple(points1_corr, points2_corr);
}

pybind11::tuple triangulatePointsLindstrom (const pybind11::array_t<double> &F_,
        const pybind11::array_t<double> &points1, const pybind11::array_t<double> &points2,
        const pybind11::array_t<bool> &good_point_mask) {
    return triangulatePointsLindstromWithK(F_, points1, points2, pybind11::none(), pybind11::none(), good_point_mask);
}

PYBIND11_MODULE(pvsac, m) {
    m.doc() = "VSAC python bindings";
    pybind11::enum_<vsac::EstimationMethod>(m, "EstimationMethod")
            .value("Homography", vsac::EstimationMethod::Homography)
            .value("Affine", vsac::EstimationMethod::Affine)
            .value("Fundamental", vsac::EstimationMethod::Fundamental)
            .value("Essential", vsac::EstimationMethod::Essential)
            .value("P3P", vsac::EstimationMethod::P3P)
            .export_values();

    pybind11::enum_<vsac::SamplingMethod>(m, "SamplingMethod")
            .value("SAMPLING_UNIFORM", vsac::SamplingMethod::SAMPLING_UNIFORM)
            .value("SAMPLING_PROSAC", vsac::SamplingMethod::SAMPLING_PROSAC)
            .export_values();

    pybind11::enum_<vsac::ScoreMethod>(m, "ScoreMethod")
            .value("SCORE_METHOD_MSAC", vsac::ScoreMethod::SCORE_METHOD_MSAC)
            .value("SCORE_METHOD_MAGSAC", vsac::ScoreMethod::SCORE_METHOD_MAGSAC)
            .export_values();

    pybind11::class_<vsac::Params> (m, "Params")
            .def(pybind11::init<vsac::EstimationMethod, double, double, int, vsac::SamplingMethod, vsac::ScoreMethod>())
            .def("setVerifier", &vsac::Params::setVerifier)
            .def("setParallel", &vsac::Params::setParallel)
            .def("setPolisher", &vsac::Params::setPolisher)
            .def("setNonRandomnessTest", &vsac::Params::setNonRandomnessTest);

    m.def("getCorrectedPointsHomography", &getCorrectedPointsHomography, "get corrected points by H", pybind11::arg("points1"), pybind11::arg("points2"),
            pybind11::arg("H"), pybind11::arg("good_point_mask"));

    m.def("triangulatePointsLindstrom", &triangulatePointsLindstrom, "get corrected points by F", pybind11::arg("F"), pybind11::arg("points1"), pybind11::arg("points2"),
            pybind11::arg("good_point_mask"));

    m.def("triangulatePointsLindstrom", &triangulatePointsLindstromWithK, "get corrected points by F, pose and 3D points", pybind11::arg("F"), pybind11::arg("points1"), pybind11::arg("points2"),
          pybind11::arg("K1"), pybind11::arg("K2"),pybind11::arg("good_point_mask"));

    m.def("estimate", &estimate, "estimate model", pybind11::arg("params"), pybind11::arg("pts1"), pybind11::arg("pts2"),
          pybind11::arg("K1")=pybind11::none(), pybind11::arg("K2")=pybind11::none(),
          pybind11::arg("dist_coef1")=pybind11::none(), pybind11::arg("dist_coef2")=pybind11::none());
}
