//#include "precomp.hpp"
//
//namespace cv { namespace vsac {
///*
// * pts1, pts2 are matrices either N x a, N x b or a x N or b x N, where N > a and N > b
// * pts1 are image points, if pnp pts2 are object points otherwise - image points as well.
// * output is matrix of size N x (a + b)
// * return points_size = N
// */
//int mergePoints (InputArray pts1_, InputArray pts2_, Mat &pts, bool ispnp) {
//    Mat pts1 = pts1_.getMat(), pts2 = pts2_.getMat();
//    auto convertPoints = [] (Mat &points, int pt_dim) {
//        points.convertTo(points, CV_32F); // convert points to have float precision
//        if (points.channels() > 1)
//            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
//        if (points.rows < points.cols)
//            transpose(points, points); // transpose so points will be in rows
//        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
//        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
//            points = points.colRange(0, pt_dim);
//    };
//
//    convertPoints(pts1, 2); // pts1 are always image points
//    convertPoints(pts2, ispnp ? 3 : 2); // for PnP points are 3D
//
//    // points are of size [Nx2 Nx2] = Nx4 for H, F, E
//    // points are of size [Nx2 Nx3] = Nx5 for PnP
//    hconcat(pts1, pts2, pts);
//    return pts.rows;
//}
//
//void saveMask (OutputArray mask, const std::vector<bool> &inliers_mask) {
//    if (mask.needed()) {
//        const int points_size = (int) inliers_mask.size();
//        mask.create(points_size, 1, CV_8U);
//        auto * maskptr = mask.getMat().ptr<uchar>();
//        for (int i = 0; i < points_size; i++)
//            maskptr[i] = (uchar) inliers_mask[i];
//    }
//}
//
//void initialize (int state, int points_size, double threshold, double max_thr, const ::vsac::Params &params, const Mat &points,
//        const Mat &calib_points, const Mat &image_points, const std::vector<Ptr<NeighborhoodGraph>> &layers, const std::vector<std::vector<int>> &close_pts_mask,
//        const Mat &K1, const Mat &K2, const Ptr<NeighborhoodGraph> &graph, Ptr<MinimalSolver> &min_solver,
//        Ptr<NonMinimalSolver> &non_min_solver, Ptr<GammaValues> &gamma_generator, Ptr<Error> &error, Ptr<Estimator> &estimator,
//        Ptr<Degeneracy> &degeneracy, Ptr<Quality> &quality,
//        Ptr<ModelVerifier> &verifier, Ptr<LocalOptimization> &lo, Ptr<Termination> &termination,
//        Ptr<Sampler> &sampler, Ptr<RandomGenerator> &lo_sampler, bool parallel_call) {
//
//#if DEBUG
//    const auto init_time = std::chrono::steady_clock::now();
//#endif
//
//    const int min_sample_size = params.getSampleSize();
//    switch (params.getError()) {
//        case ::vsac::ErrorMetric::SYMM_REPR_ERR:
//            error = ReprojectionErrorSymmetric::create(points); break;
//        case ::vsac::ErrorMetric::FORW_REPR_ERR:
//            if (params.getEstimator() == ::vsac::EstimationMethod::Affine)
//                error = ReprojectionErrorAffine::create(points);
//            else error = ReprojectionErrorForward::create(points);
//            break;
//        case ::vsac::ErrorMetric::SAMPSON_ERR:
//            error = SampsonError::create(points); break;
//        case ::vsac::ErrorMetric::SGD_ERR:
//            error = SymmetricGeometricDistance::create(points); break;
//        case ::vsac::ErrorMetric::RERPOJ:
//            error = ReprojectionErrorPmatrix::create(points); break;
//        default: CV_Error(cv::Error::StsNotImplemented , "Error metric is not implemented!");
//    }
//
//    if (params.getScore() == ::vsac::ScoreMethod::SCORE_METHOD_MAGSAC || params.getLO() == ::vsac::LocalOptimMethod::LOCAL_OPTIM_SIGMA)
//        gamma_generator = GammaValues::create(params.getDegreesOfFreedom());
//
//    const double k_mlesac = params.getKmlesac ();
//    switch (params.getScore()) {
//        case ::vsac::ScoreMethod::SCORE_METHOD_RANSAC :
//            quality = RansacQuality::create(points_size, threshold, error); break;
//        case ::vsac::ScoreMethod::SCORE_METHOD_MSAC :
//            quality = MsacQuality::create(points_size, threshold, error, k_mlesac); break;
//        case ::vsac::ScoreMethod::SCORE_METHOD_MAGSAC :
//            quality = MagsacQuality::create(max_thr, points_size, error, gamma_generator,
//                threshold, params.getDegreesOfFreedom(),  params.getSigmaQuantile(),
//                params.getUpperIncompleteOfSigmaQuantile(),
//                params.getLowerIncompleteOfSigmaQuantile(), params.getC()); break;
//        case ::vsac::ScoreMethod::SCORE_METHOD_LMEDS :
//            quality = LMedsQuality::create(points_size, threshold, error); break;
//        default: CV_Error(cv::Error::StsNotImplemented, "Score is not imeplemeted!");
//    }
//
//    if (params.isHomography()) {
//        degeneracy = HomographyDegeneracy::create(points);
//        if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
//            min_solver = HomographySVDSolver::create(points);
//        else min_solver = HomographyMinimalSolver4ptsGEM::create(points);
//        non_min_solver = HomographyNonMinimalSolver::create(points);
//        estimator = HomographyEstimator::create(min_solver, non_min_solver, degeneracy);
//    } else if (params.isFundamental()) {
//        degeneracy = FundamentalDegeneracy::create(state++, quality, points, min_sample_size,
//               params.getPlaneAndParallaxIters(), 8. /*sqr homogr thr*/, K1, K2);
//        degeneracy->setClosePointsMask(close_pts_mask);
//        const auto img_size = params.getImageSize();
//        if (K1.empty() && img_size.width != 0 && img_size.height != 0) {
//            if (img_size.width > img_size.height)
//                 degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(img_size.width/2., img_size.height/2.);
//            else degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(img_size.height/2., img_size.width/2.);
//        }
//        if(min_sample_size == 7) {
//            if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
//                min_solver = FundamentalSVDSolver::create(points);
//            else min_solver = FundamentalMinimalSolver7pts::create(points);
//        } else min_solver = FundamentalMinimalSolver8pts::create(points);
//        non_min_solver = FundamentalNonMinimalSolver::create(points);
//        estimator = FundamentalEstimator::create(min_solver, non_min_solver, degeneracy);
//    } else if (params.isEssential()) {
//        degeneracy = EssentialDegeneracy::create(points, min_sample_size);
//        if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
//            min_solver = EssentialMinimalSolverStewenius5ptsSVD::create(points);
//        else min_solver = EssentialMinimalSolverStewenius5pts::create(points);
////        std::cout << points << "\n";
//        non_min_solver = EssentialNonMinimalSolver::create(points);
////        non_min_solver = EssentialNonMinimalSolverViaF::create(image_points, K1, K2);
////        non_min_solver = EssentialNonMinimalSolverViaT::create(points);
//        estimator = EssentialEstimator::create(min_solver, non_min_solver, degeneracy);
//    } else if (params.isPnP()) {
//        degeneracy = makePtr<Degeneracy>();
//        if (min_sample_size == 3) {
//            min_solver = P3PSolver::create(points, calib_points, K1);
//            non_min_solver = DLSPnP::create(points, calib_points, K1);
////            non_min_solver = PnPNonMinimalSolver::create(points);
//        } else {
//            if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
//                min_solver = PnPSVDSolver::create(points);
//            else min_solver = PnPMinimalSolver6Pts::create(points);
//            non_min_solver = PnPNonMinimalSolver::create(points);
//        }
//        estimator = PnPEstimator::create(min_solver, non_min_solver);
//    } else if (params.getEstimator() == ::vsac::EstimationMethod::Affine) {
//        degeneracy = makePtr<Degeneracy>();
//        min_solver = AffineMinimalSolver::create(points);
//        non_min_solver = AffineNonMinimalSolver::create(points);
//        estimator = AffineEstimator::create(min_solver, non_min_solver);
//    } else CV_Error(cv::Error::StsNotImplemented, "Estimator not implemented!");
//
//    switch (params.getSampler()) {
//        case ::vsac::SamplingMethod::SAMPLING_UNIFORM:
//            if (params.isQuasiSampling())
//                sampler = QuasiUniformSampler::create(state++, min_sample_size, points_size);
//            else sampler = UniformSampler::create(state++, min_sample_size, points_size);
//            break;
//        case ::vsac::SamplingMethod::SAMPLING_PROSAC:
//            if (!parallel_call) // for parallel only one PROSAC sampler
//                sampler = ProsacSampler::create(state++, points_size, min_sample_size, 200000);
//            break;
//        case ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC:
//            sampler = ProgressiveNapsac::create(state++, points_size, min_sample_size, layers, 20); break;
//        case ::vsac::SamplingMethod::SAMPLING_NAPSAC:
//            sampler = NapsacSampler::create(state++, points_size, min_sample_size, graph); break;
//        default: CV_Error(cv::Error::StsNotImplemented, "Sampler is not implemented!");
//    }
//
//    switch (params.getVerifier()) {
//        case ::vsac::VerificationMethod::NullVerifier: verifier = ModelVerifier::create(); break;
//        case ::vsac::VerificationMethod::SprtVerifier:
//            verifier = AdaptiveSPRT::create(state++, error, quality, points_size, params.getScore() == ::vsac::ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
//             params.getSPRTepsilon(), params.getSPRTdelta(), params.getTimeForModelEstimation(),
//             params.getSPRTavgNumModels(), params.getScore(), params.isParallel() ? 0 : params.getMaxNumHypothesisToTestBeforeRejection(), k_mlesac, false); break;
//        case ::vsac::VerificationMethod::ASPRT:
//            verifier = AdaptiveSPRT::create(state++, error, quality, points_size, params.getScore() == ::vsac::ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
//             params.getSPRTepsilon(), params.getSPRTdelta(), params.getTimeForModelEstimation(),
//             params.getSPRTavgNumModels(), params.getScore(), params.isParallel() ? 0 : params.getMaxNumHypothesisToTestBeforeRejection(), k_mlesac); break;
//        default: CV_Error(cv::Error::StsNotImplemented, "Verifier is not imeplemented!");
//    }
//
//    if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROSAC) {
//        if (!parallel_call)
//            termination = ProsacTerminationCriteria::create(sampler.dynamicCast<ProsacSampler>(), error,
//                points_size, min_sample_size, params.getConfidence(), params.getMaxIters(), 100, 0.05, 0.05, threshold);
//    } else if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
//        if (params.getVerifier() == ::vsac::VerificationMethod::SprtVerifier)
//            termination = SPRTPNapsacTermination::create(((AdaptiveSPRT *)verifier.get())->getSPRTvector(),
//                    params.getConfidence(), points_size, min_sample_size,
//                    params.getMaxIters(), params.getRelaxCoef());
//        else
//            termination = StandardTerminationCriteria::create (params.getConfidence(),
//                    points_size, min_sample_size, params.getMaxIters());
//    } else if (params.getVerifier() == ::vsac::VerificationMethod::SprtVerifier) {
//        termination = SPRTTermination::create(((AdaptiveSPRT *) verifier.get())->getSPRTvector(),
//             params.getConfidence(), points_size, min_sample_size, params.getMaxIters());
//    } else
//        termination = StandardTerminationCriteria::create
//            (params.getConfidence(), points_size, min_sample_size, params.getMaxIters());
//
//    if (params.getLO() != ::vsac::LocalOptimMethod::LOCAL_OPTIM_NULL) {
//        lo_sampler = UniformRandomGenerator::create(state, points_size, params.getLOSampleSize());
//        const bool force_LO = params.isForceLO();
//        const auto lo_termination = StandardTerminationCriteria::create(params.getConfidence(), points_size, min_sample_size, params.getMaxIters());
//        switch (params.getLO()) {
//            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_INNER_LO:
//                lo = SimpleLocalOptimization::create(degeneracy, quality, estimator, force_LO ? nullptr : lo_termination, lo_sampler,
//                                         params.getLOInnerMaxIters()); break;
//            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_INNER_AND_ITER_LO:
//                lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler,
//                     points_size, threshold, true, params.getLOIterativeSampleSize(),
//                     params.getLOInnerMaxIters(), params.getLOIterativeMaxIters(),
//                     params.getLOThresholdMultiplier()); break;
//            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_GC:
//                lo = GraphCut::create(estimator, error, quality, graph, lo_sampler, threshold,
//                   params.getGraphCutSpatialCoherenceTerm(), params.getLOInnerMaxIters(), force_LO ? nullptr : lo_termination); break;
//            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_SIGMA:
//                lo = SigmaConsensus::create(estimator, error, quality, verifier, gamma_generator,
//                     params.getLOSampleSize(), params.getLOInnerMaxIters(),
//                     params.getDegreesOfFreedom(), params.getSigmaQuantile(),
//                     params.getUpperIncompleteOfSigmaQuantile(), params.getC(), max_thr, force_LO ? nullptr : lo_termination); break;
//            default: CV_Error(cv::Error::StsNotImplemented , "Local Optimization is not implemented!");
//        }
//    }
//#if DEBUG
//    std::cout << "Init time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - init_time).count() << '\n';
//#endif
//}
//
//int getNumberOfNonRandomInliers (double scale, const Mat &points, bool is_F, const Mat &model,
//        bool has_sample, const std::vector<int> &sample, const std::vector<int> &inliers_, const int num_inliers_) {
//    int num_inliers = num_inliers_;
//    if (num_inliers == 0) return 0;
//    std::vector<int> inliers= inliers_;
//    const auto * const pts = (float *) points.data;
//     const double ep_thr_sqr = 1e-6, line_thr = scale*0.01, /*5*/ neigh_thr_sqr = scale * 10; //scale * 9, 50
////    const double ep_thr_sqr = 1e-5, line_thr = scale*0.1, /*5*/ neigh_thr_sqr = scale * 100; //scale * 9, 50
//    int  num_pts_bad_conditioning = 0, num_pts_near_ep = 0,
//        num_pts_on_ep_lines = 0, num_pts_validatin_or_constr = 0, pt1 = 0;
//    const int sample_size = is_F ? 7 : 4;
//    double sign1 = 0, a1=0, b1=0, c1=0, a2=0, b2=0, c2=0;
//    const auto * const m = (double *) model.data;
//    Vec3d ep1;
//    double ep1_x, ep1_y, ep2_x, ep2_y;
//    bool do_or_test = false, ep1_inf = false, ep2_inf = false;
//    if (is_F) { // compute epipole and sign of the first point for orientation test
//        ep1 = model.row(0).cross(model.row(2));
//        auto * e = ep1.val;
//
//        // e is zero vector, recompute e
//        if (e[0] <= 1.9984e-15 && e[0] >= -1.9984e-15 &&
//            e[1] <= 1.9984e-15 && e[1] >= -1.9984e-15 &&
//            e[2] <= 1.9984e-15 && e[2] >= -1.9984e-15) {
//            ep1 = model.row(1).cross(model.row(2));
//        }
//
//        cv::Vec3d ep2 = model.col(0).cross(model.col(2));
//        e = ep2.val;
//
//        // e is zero vector, recompute e
//        if (e[0] <= 1.9984e-15 && e[0] >= -1.9984e-15 &&
//            e[1] <= 1.9984e-15 && e[1] >= -1.9984e-15 &&
//            e[2] <= 1.9984e-15 && e[2] >= -1.9984e-15) {
//            ep2 = model.col(1).cross(model.col(2));
//        }
//
//        if (fabs(ep1[2]) < DBL_EPSILON) {
//            ep1_inf = true; ep1_x = DBL_MAX; ep1_y = DBL_MAX;
//        } else {
//            ep1_x = ep1[0] / ep1[2];
//            ep1_y = ep1[1] / ep1[2];
//        }
//        if (fabs(ep2[2]) < DBL_EPSILON) {
//            ep2_inf = true; ep2_x = DBL_MAX; ep2_y = DBL_MAX;
//        } else {
//            ep2_x = ep2[0] / ep2[2];
//            ep2_y = ep2[1] / ep2[2];
//        }
//    }
//    const auto * const e1 = ep1.val; // of size 3x1
//
//    // we move sample points to the end, so every inlier will be checked by sample point
//    int num_sample_in_inliers;
//    if (has_sample) {
//        num_sample_in_inliers = 0;
//        int temp_idx = num_inliers;
//        for (int i = 0; i < temp_idx; i++) {
//            const int inl = inliers[i];
//            for (int s : sample) {
//                if (inl == s) {
//                    std::swap(inliers[i], inliers[--temp_idx]);
//                    i--; // we need to check inlier that we just swapped
//                    num_sample_in_inliers++;
//                    break;
//                }
//            }
//        }
//        if (num_sample_in_inliers < sample_size) {
//            num_sample_in_inliers = sample_size; // for good model a few points does not matter, for bad could be decisive
//        }
//    } else num_sample_in_inliers = sample_size;
//
//    if (is_F) {
//        int MIN_TEST = std::min(15, num_inliers);
//        for (int i = 0; i < MIN_TEST; i++) {
//            pt1 = 4*inliers[i];
//            sign1 = (m[0]*pts[pt1+2]+m[3]*pts[pt1+3]+m[6])*(e1[1]-e1[2]*pts[pt1+1]);
//            int validate = 0;
//            for (int j = 0; j < MIN_TEST; j++) {
//                if (i == j) continue;
//                const int inl_idx = 4*inliers[j];
//                if (sign1*(m[0]*pts[inl_idx+2]+m[3]*pts[inl_idx+3]+m[6])*(e1[1]-e1[2]*pts[inl_idx+1])<0) {
//                    validate++;
//                }
//            }
//            if (validate < MIN_TEST/2) {
//                do_or_test = true;
//                break;
//            }
//        }
//    }
//
//
//    // verification does not include sample points as they surely random
//    const int max_verify = num_inliers - num_sample_in_inliers;
//    if (max_verify <= 0)
//        return 0;
//    int num_non_random_inliers = max_verify;
//    auto removeDependentPoints = [&] (bool do_orient_test, bool check_epipoles) {
//        for (int i = 0; i < max_verify; i++) {
//            // checks over inliers if they are dependent to other inliers
//            const int inl_idx = 4*inliers[i];
//            const double x1 = pts[inl_idx], y1 = pts[inl_idx+1], x2 = pts[inl_idx+2], y2 = pts[inl_idx+3];
//            if (is_F) {
//                // epipolar line on image 2 = l2
//                a2 = m[0] * x1 + m[1] * y1 + m[2];
//                b2 = m[3] * x1 + m[4] * y1 + m[5];
//                c2 = m[6] * x1 + m[7] * y1 + m[8];
//                // epipolar line on image 1 = l1
//                a1 = m[0] * x2 + m[3] * y2 + m[6];
//                b1 = m[1] * x2 + m[4] * y2 + m[7];
//                c1 = m[2] * x2 + m[5] * y2 + m[8];
////            std::cout << a2 * a2 + b2 * b2 + c2 * c2 << " " << a1 * a1 + b1 * b1 + c1 * c1 << " dist ep " << (x1-ep1_x)*(x1-ep1_x)+(y1-ep1_y)*(y1-ep1_y) << " " <<  (x2-ep2_x)*(x2-ep2_x)+(y2-ep2_y)*(y2-ep2_y)<< "\n";
//                if ((!ep1_inf && (x1-ep1_x)*(x1-ep1_x)+(y1-ep1_y)*(y1-ep1_y) < neigh_thr_sqr) ||
//                    (!ep2_inf && (x2-ep2_x)*(x2-ep2_x)+(y2-ep2_y)*(y2-ep2_y) < neigh_thr_sqr)) {
////                std::cout << "inl " << inliers[i] << "near ep\n";
//                    num_non_random_inliers--;
//                    num_pts_near_ep++;
//                    continue; // is dependent, continue to the next point
//                } else if (check_epipoles) {
//                    if (a2 * a2 + b2 * b2 + c2 * c2 < ep_thr_sqr ||
//                        a1 * a1 + b1 * b1 + c1 * c1 < ep_thr_sqr) {
//                        num_non_random_inliers--;
//                        num_pts_near_ep++;
//                        continue; // is dependent, continue to the next point
//                    }
//                }
//                else if (do_orient_test && pt1 != inl_idx && sign1*(m[0]*x2+m[3]*y2+m[6])*(e1[1]-e1[2]*y1)<0) {
//                    num_non_random_inliers--;
//                    num_pts_validatin_or_constr++;
//                    continue;
//                }
//                const auto mag2 = 1 / sqrt(a2 * a2 + b2 * b2), mag1 = 1/sqrt(a1 * a1 + b1 * b1);
//                a2 *= mag2; b2 *= mag2; c2 *= mag2;
//                a1 *= mag1; b1 *= mag1; c1 *= mag1;
//            }
//
//            for (int j = i+1; j < num_inliers; j++) {// verify through all including sample points
//                const int inl_idx_j = 4*inliers[j];
//                const double X1 = pts[inl_idx_j], Y1 = pts[inl_idx_j+1], X2 = pts[inl_idx_j+2], Y2 = pts[inl_idx_j+3];
//                const double dx1 = X1-x1, dy1 = Y1-y1, dx2 = X2-x2, dy2 = Y2-y2;
////            std::cout << inliers[i] << " to " << inliers[j] << " dists "<< dx1 * dx1 + dy1 * dy1 << " " << dx2 * dx2 + dy2 * dy2 << "\n";
//                if (dx1 * dx1 + dy1 * dy1 < neigh_thr_sqr || dx2 * dx2 + dy2 * dy2 < neigh_thr_sqr) {
//                    num_non_random_inliers--;
//                    num_pts_bad_conditioning++;
//                    break; // is dependent stop verification
//                } else if (is_F) {
//                    if (fabs(a2 * X2 + b2 * Y2 + c2) < line_thr && //|| // xj'^T F   xi
//                        fabs(a1 * X1 + b1 * Y1 + c1) < line_thr) { // xj^T  F^T xi'
//                        num_non_random_inliers--;
//                        num_pts_on_ep_lines++;
//                        break; // is dependent stop verification
//                    }
//                }
//            }
//        }
//    };
//    removeDependentPoints(do_or_test, !ep1_inf && !ep2_inf);
//    const bool is_pts_vald_constr_normal = (double)num_pts_validatin_or_constr / num_inliers < 0.6;
//    const bool is_pts_near_ep_normal = (double)num_pts_near_ep / num_inliers < 0.6;
//    if (!is_pts_near_ep_normal || !is_pts_vald_constr_normal) {
////        std::cout << "Something not normal " << is_pts_vald_constr_normal << " " <<is_pts_near_ep_normal <<", recompute\n";
////    std::cout << "Before: non-rand " << num_non_random_inliers << " pts valid or constr " << num_pts_validatin_or_constr << " near ep " << num_pts_near_ep << " close " << num_pts_bad_conditioning
////              << " on ep lines " << num_pts_on_ep_lines << " in sample " << num_sample_in_inliers << " input inliers " <<num_inliers << "\n";
//        num_non_random_inliers = max_verify;
//        num_pts_bad_conditioning = 0; num_pts_near_ep = 0; num_pts_on_ep_lines = 0; num_pts_validatin_or_constr = 0;
//        removeDependentPoints(is_pts_vald_constr_normal, is_pts_near_ep_normal);
////    std::cout << "After: non-rand " << num_non_random_inliers << " pts valid or constr " << num_pts_validatin_or_constr << " near ep " << num_pts_near_ep << " close " << num_pts_bad_conditioning
////              << " on ep lines " << num_pts_on_ep_lines << " in sample " << num_sample_in_inliers << " input inliers " <<num_inliers << "\n";
//    }
//    return num_non_random_inliers;
//}
//
//}}
//
//namespace vsac {
//bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
//               Output &output, cv::InputArray K1_, cv::InputArray K2_,
//               cv::InputArray dist_coeff1, cv::InputArray dist_coeff2) {
//    cv::Ptr<cv::vsac::NeighborhoodGraph> graph;
//    cv::Ptr<cv::vsac::Error> error;
//    cv::Ptr<cv::vsac::Estimator> estimator;
//    cv::Ptr<cv::vsac::Degeneracy> degeneracy;
//    cv::Ptr<cv::vsac::Quality> quality;
//    cv::Ptr<cv::vsac::ModelVerifier> verifier;
//    cv::Ptr<cv::vsac::Sampler> sampler;
//    cv::Ptr<cv::vsac::RandomGenerator> lo_sampler;
//    cv::Ptr<cv::vsac::Termination> termination;
//    cv::Ptr<cv::vsac::LocalOptimization> lo;
//    cv::Ptr<cv::vsac::FinalModelPolisher> polisher;
//    cv::Ptr<cv::vsac::MinimalSolver> min_solver;
//    cv::Ptr<cv::vsac::NonMinimalSolver> non_min_solver;
//    cv::Ptr<cv::vsac::GammaValues> gamma_generator;
//
//    int state = params.getRandomGeneratorState();
//    cv::Mat points, K1, K2, calib_points, undist_points1, undist_points2, image_points;
//    int points_size;
//    double threshold = params.getThreshold(), max_thr = params.getMaximumThreshold();
//    if (params.isPnP()) {
//        if (! K1_.empty()) {
//            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
//            if (! dist_coeff1.empty()) {
//                // undistortPoints also calibrate points using K
//                if (points1.isContinuous())
//                    undistortPoints(points1, undist_points1, K1_, dist_coeff1);
//                else undistortPoints(points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
//                points_size = cv::vsac::mergePoints(undist_points1, points2, points, true);
//                cv::vsac::Utils::normalizeAndDecalibPointsPnP (K1, points, calib_points);
//            } else {
//                points_size = cv::vsac::mergePoints(points1, points2, points, true);
//                cv::vsac::Utils::calibrateAndNormalizePointsPnP(K1, points, calib_points);
//            }
//        } else points_size = cv::vsac::mergePoints(points1, points2, points, true);
//    } else {
//        if (params.isEssential()) {
//            CV_CheckEQ((int)(!K1_.empty() && !K2_.empty()), 1, "Intrinsic matrix must not be empty!");
//            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
//            K2 = K2_.getMat(); K2.convertTo(K2, CV_64F);
//            if (! dist_coeff1.empty() || ! dist_coeff2.empty()) {
//                // undistortPoints also calibrate points using K
//                if (points1.isContinuous())
//                    undistortPoints(points1, undist_points1, K1_, dist_coeff1);
//                else undistortPoints(points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
//                if (points2.isContinuous())
//                    undistortPoints(points2, undist_points2, K2_, dist_coeff2);
//                else undistortPoints(points2.getMat().clone(), undist_points2, K2_, dist_coeff2);
//                points_size = cv::vsac::mergePoints(undist_points1, undist_points2, calib_points, false);
//            } else {
//                points_size = cv::vsac::mergePoints(points1, points2, points, false);
//                cv::vsac::Utils::calibratePoints(K1, K2, points, calib_points);
//            }
//            threshold = cv::vsac::Utils::getCalibratedThreshold(threshold, K1, K2);
//            max_thr = cv::vsac::Utils::getCalibratedThreshold(max_thr, K1, K2);
//        } else {
//            if (params.isFundamental()) {
//                K1 = K1_.getMat(); K2 = K2_.getMat();
//                if (! K1.empty() && ! K2.empty()) {
//                    K1.convertTo(K1, CV_64F); K2.convertTo(K2, CV_64F);
//                }
//            }
//            points_size = cv::vsac::mergePoints(points1, points2, points, false);
//        }
//    }
//
//    // Since error function output squared error distance, so make
//    // threshold squared as well
//    threshold *= threshold;
//
//    //todo: add inner_thre
//    if (params.getSampler() == SamplingMethod::SAMPLING_NAPSAC || params.getLO() == LocalOptimMethod::LOCAL_OPTIM_GC) {
//        if (params.getNeighborsSearch() == NeighborSearchMethod::NEIGH_GRID) {
//            graph = cv::vsac::GridNeighborhoodGraph::create(points, points_size,
//                    params.getCellSize(), params.getCellSize(), params.getCellSize(), params.getCellSize(), 10);
//        } else if (params.getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_KNN) {
//            graph = cv::vsac::FlannNeighborhoodGraph::create(points, points_size,params.getKNN(), false, 5, 1);
//        } else if (params.getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_RADIUS) {
//            assert(false && "check if it RadiusSearchNeighborhoodGraph works properly!");
//            graph = cv::vsac::RadiusSearchNeighborhoodGraph::create(points, points_size,
//                                                                    params.getGraphRadius(), 5, 1);
//        } else CV_Error(cv::Error::StsNotImplemented, "Graph type is not implemented!");
//    }
//
//    std::vector<cv::Ptr<cv::vsac::NeighborhoodGraph>> layers;
//    if (params.getSampler() == SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
//        CV_CheckEQ((int)params.isPnP(), 0, "ProgressiveNAPSAC for PnP is not implemented!");
//        const auto &cell_number_per_layer = params.getGridCellNumber();
//        layers.reserve(cell_number_per_layer.size());
//        const auto * const pts = (float *) points.data;
//        float img1_width = 0, img1_height = 0, img2_width = 0, img2_height = 0;
//        for (int i = 0; i < 4 * points_size; i += 4) {
//            if (pts[i    ] > img1_width ) img1_width  = pts[i    ];
//            if (pts[i + 1] > img1_height) img1_height = pts[i + 1];
//            if (pts[i + 2] > img2_width ) img2_width  = pts[i + 2];
//            if (pts[i + 3] > img2_height) img2_height = pts[i + 3];
//        }
//        // Create grid graphs (overlapping layes of given cell numbers)
//        for (int layer_idx = 0; layer_idx < (int)cell_number_per_layer.size(); layer_idx++) {
//            const int cell_number = cell_number_per_layer[layer_idx];
//            if (layer_idx > 0)
//                if (cell_number_per_layer[layer_idx-1] <= cell_number)
//                    CV_Error(cv::Error::StsError, "Progressive NAPSAC sampler: "
//                                                  "Cell number in layers must be in decreasing order!");
//            layers.emplace_back(cv::vsac::GridNeighborhoodGraph::create(points, points_size,
//        (int)(img1_width / (float)cell_number), (int)(img1_height / (float)cell_number),
//        (int)(img2_width / (float)cell_number), (int)(img2_height / (float)cell_number), 10));
//        }
//    }
//
//    // update points by calibrated for Essential matrix after graph is calculated
//    if (params.isEssential()) {
//        points.copyTo(image_points);
//        points = calib_points;
//        // if maximum calibrated threshold significanlty differs threshold then set upper bound
//        if (max_thr > 10*threshold)
//            max_thr = sqrt(10*threshold); // max thr will be squared after
//    }
//    if (max_thr < threshold)
//        max_thr = threshold;
//
//    std::vector<std::vector<int>> close_pts_mask;
//    initialize (state, points_size, threshold, max_thr, params, points, calib_points, image_points, layers, close_pts_mask, K1, K2, graph, min_solver,
//                non_min_solver, gamma_generator, error, estimator, degeneracy, quality, verifier, lo, termination, sampler, lo_sampler, false);
//
//    cv::Ptr<cv::vsac::RandomGenerator> polisher_sampler;
//    cv::Ptr<cv::vsac::CovarianceSolver> cov_polisher;
//    switch (params.getFinalPolisher()) {
//        case PolishingMethod::CovPolisher:
//            if (params.isFundamental()) {
//                cov_polisher = cv::vsac::CovarianceFundamentalSolver::create(points);
//            } else if (params.isHomography()) {
//                cov_polisher = cv::vsac::CovarianceHomographySolver::create(points);
//            } else if (params.isEssential()) {
//                cov_polisher = cv::vsac::CovarianceEssentialSolver::create(calib_points);
//                if (! params.isEnforceRank())
//                    cov_polisher->setEnforceRankConstraint(false);
//            } else if (params.getEstimator() == EstimationMethod::Affine || params.isPnP()) {
//                polisher = cv::vsac::LeastSquaresPolishing::create(estimator, quality, params.getFinalLSQIterations()); // use lsq polisher here
//                break;
//            } else assert(false && "covariance polisher not implemented\n");
//            polisher = cv::vsac::CovariancePolisher::create(degeneracy, quality, cov_polisher, params.getFinalLSQIterations());
//            break;
//        case PolishingMethod::MAGSAC:
//            if (gamma_generator == nullptr)
//                gamma_generator = cv::vsac::GammaValues::create(params.getDegreesOfFreedom());
//            polisher = cv::vsac::WeightedPolisher::create(degeneracy, quality, non_min_solver, quality->getErrorFnc(), gamma_generator, 10, params.getDegreesOfFreedom(), params.getUpperIncompleteOfSigmaQuantile(),
//                                                          params.getC(), 2*max_thr);
//            break;
//        case PolishingMethod ::LSQPolisher:
//            polisher = cv::vsac::LeastSquaresPolishing::create(estimator, quality, params.getFinalLSQIterations());
//            break;
//        case PolishingMethod ::IterativePolish:
//            polisher_sampler = cv::vsac::UniformRandomGenerator::create(state++, points_size, params.getLOSampleSize());
//            polisher = cv::vsac::IterativePolisher::create(quality, degeneracy, estimator, polisher_sampler, 20);
//            break;
//        default : break;
//    }
//
//    const bool is_parallel = params.isParallel();
//    cv::vsac::VSAC vsac (points, params, points_size, estimator, quality, sampler,
//                           termination, verifier, degeneracy, lo, polisher, is_parallel, state);
//
//    if (is_parallel)
//        vsac.setDataForParallel (threshold, max_thr, K1, K2, calib_points, image_points, graph, layers, close_pts_mask);
//    return vsac.run(output);
//}
//}
