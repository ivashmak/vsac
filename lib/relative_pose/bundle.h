// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_BUNDLE_H_
#define POSELIB_BUNDLE_H_

#include <opencv2/core.hpp>

namespace pose_lib {
struct CameraPose {
    cv::Matx33d R;
    cv::Vec3d t;
    double alpha = 1.0; // either focal length or scale
};
typedef std::vector<CameraPose> CameraPoseVector;

struct BundleOptions {
    int max_iterations = 100;
    enum LossType {
        MLESAC,
    } loss_type = LossType::MLESAC; // CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-8;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
};

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_relpose(const cv::Mat &correspondences_,
                   const std::vector<int> &sample_,
                   const int sample_size_,
                   CameraPose *pose,
                   const BundleOptions &opt = BundleOptions(),
                   const double *weights = nullptr);
} // namespace pose_lib

#endif