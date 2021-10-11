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

#include "precomp.hpp"

namespace cv { namespace vsac {
bool Utils::satisfyCheirality (const cv::Matx33d& R, const cv::Vec3d &t, const cv::Vec3d &x1, const cv::Vec3d &x2) {
    // This code assumes that x1 and x2 are unit vectors
     const auto Rx1 = R * x1;
    // lambda_2 * x2 = R * ( lambda_1 * x1 ) + t
    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
    const double a = -Rx1.dot(x2), b1 = -Rx1.dot(t), b2 = x2.dot(t);
    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    return (b1 - a * b2 > 0) && (-a * b1 + b2 > 0);
}
}}