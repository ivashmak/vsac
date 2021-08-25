#ifndef PRECOMP_HPP
#define PRECOMP_HPP

// C++
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <map>
#include <exception>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

// VSAC module
#include "../include/vsac.hpp"

#if defined(HAVE_EIGEN)
#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>
#else
#define HAVE_EIGEN
#include "../lib/Eigen/Eigen"
#include "../lib/Eigen/src/MatrixFunctions/MatrixSquareRoot.h"
#endif

#if defined(HAVE_LAPACK)
#include <lapacke.h>
#endif

#endif // PRECOMP_HPP
