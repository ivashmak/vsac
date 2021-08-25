#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "../include/vsac.hpp"

pybind11::tuple estimate(const vsac::Params &params,
        const pybind11::array_t<double> &pts1, const pybind11::array_t<double> &pts2,
        pybind11::array_t<double> K1, pybind11::array_t<double> K2,
        pybind11::array_t<double> d_coef1, pybind11::array_t<double> d_coef2) {

    const int points_size = pts1.request().shape[0];
    const int dim1 = pts1.request().shape[1], dim2 = pts2.request().shape[1];
    const int num_cols = (int) (pts1.request().strides[0] / pts1.request().strides[1]);
    cv::Mat points1, points2;
    if (num_cols == 4 || num_cols == 5) {
        cv::Mat points (points_size, num_cols, CV_64F, (double*)pts1.request().ptr);
        points.colRange(0,2).copyTo(points1);
        points.colRange(2, num_cols).copyTo(points2);
    } else if (num_cols == 2) {
        points1 = cv::Mat(points_size, dim1, CV_64F, (double*)pts1.request().ptr);
        points2 = cv::Mat(points_size, dim2, CV_64F, (double*)pts2.request().ptr);
    } else {
        std::cerr << "Incorrect size of points!\n";
        exit(1);
    }

    cv::Mat distortion_coeff1, distortion_coeff2, _K1, _K2;
    if (K1.size() != 1) _K1 = cv::Mat_<double>(3,3, (double *)K1.request().ptr);
    if (K2.size() != 1) _K2 = cv::Mat_<double>(3,3, (double *)K2.request().ptr);
    if (d_coef1.size() != 1) distortion_coeff1 = cv::Mat_<double>(d_coef1.request().shape[0],d_coef1.request().shape[1], (double *)d_coef1.request().ptr);
    if (d_coef2.size() != 1) distortion_coeff2 = cv::Mat_<double>(d_coef2.request().shape[0],d_coef2.request().shape[1], (double *)d_coef2.request().ptr);

    pybind11::array_t<double> model_out;
    pybind11::array_t<bool> inliers_out;
    vsac::Output output;
    if (vsac::estimate(params, points1, points2, output, _K1, _K2, distortion_coeff1, distortion_coeff2)) {
        const auto &model = output.getModel();
        // copy model
        model_out = pybind11::array_t<double>({model.rows, model.cols});
        std::copy((double*)model.data, (double*)model.data+model.rows*model.cols, (double*)model_out.request().ptr);

        if (params.isMaskRequired()) {
            const auto &inliers_mask = output.getInliersMask();
            // copy inliers
            inliers_out = pybind11::array_t<bool>(points_size);
            std::copy(inliers_mask.begin(), inliers_mask.end(), (bool*)inliers_out.request().ptr);
        }
    }
    return pybind11::make_tuple(model_out, inliers_out);
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
            .def(pybind11::init<double , vsac::EstimationMethod, vsac::SamplingMethod, double, int, vsac::ScoreMethod>());

    m.def("estimate", &estimate, "estimate model", pybind11::arg("params"), pybind11::arg("pts1"), pybind11::arg("pts2"),
            pybind11::arg("K1")=pybind11::none(), pybind11::arg("K2")=pybind11::none(),
          pybind11::arg("dist_coef1")=pybind11::none(), pybind11::arg("dist_coef2")=pybind11::none());
}
