#include "precomp.hpp"

namespace cv {namespace vsac {
void VSAC::initialize (int state, Ptr<MinimalSolver> &min_solver, Ptr<NonMinimalSolver> &non_min_solver, Ptr<GammaValues> &gamma_generator,
        Ptr<Error> &error_sample, Ptr<Error> &error_val, Ptr<Estimator> &estimator, Ptr<Degeneracy> &degeneracy, Ptr<Quality> &quality,
        Ptr<ModelVerifier> &verifier, Ptr<LocalOptimization> &lo, Ptr<Termination> &termination,
        Ptr<Sampler> &sampler, Ptr<RandomGenerator> &lo_sampler, Ptr<Termination> &lo_termination, bool parallel_call) {
#ifdef DEBUG
    const auto init_time = std::chrono::steady_clock::now();
#endif
    const int min_sample_size = params.getSampleSize();

    // inner inlier threshold will be used in LO to obtain inliers
    // additionally in DEGENSAC for F
    double inner_inlier_thr_sqr = threshold;
    if (params.isHomography() && inner_inlier_thr_sqr < 5.25) inner_inlier_thr_sqr = 5.25; // at least 2.5 px
    else if (params.isFundamental() && inner_inlier_thr_sqr < 4) inner_inlier_thr_sqr = 4; // at least 2 px

    switch (params.getError()) {
        case ::vsac::ErrorMetric::SYMM_REPR_ERR:
            error_val = ReprojectionErrorSymmetric::create(points_val);
            error_sample = ReprojectionErrorSymmetric::create(points_sample);
            break;
        case ::vsac::ErrorMetric::FORW_REPR_ERR:
            if (params.getEstimator() == ::vsac::EstimationMethod::Affine) {
                error_val = ReprojectionErrorAffine::create(points_val);
                error_sample = ReprojectionErrorAffine::create(points_sample);
            } else {
                error_val = ReprojectionErrorForward::create(points_val);
                error_sample = ReprojectionErrorForward::create(points_sample);
            }
            break;
        case ::vsac::ErrorMetric::SAMPSON_ERR:
            error_val = SampsonError::create(points_val);
            error_sample = SampsonError::create(points_sample);
            break;
        case ::vsac::ErrorMetric::SGD_ERR:
            error_val = SymmetricGeometricDistance::create(points_val);
            error_sample = SymmetricGeometricDistance::create(points_sample);
            break;
        case ::vsac::ErrorMetric::RERPOJ:
            error_val = ReprojectionErrorPmatrix::create(points_val);
            error_sample = ReprojectionErrorPmatrix::create(points_sample);
            break;
        default: CV_Error(cv::Error::StsNotImplemented , "Error metric is not implemented!");
    }

    if (params.getScore() == ::vsac::ScoreMethod::SCORE_METHOD_MAGSAC || params.getLO() == ::vsac::LocalOptimMethod::LOCAL_OPTIM_SIGMA)
        gamma_generator = GammaValues::create(params.getDegreesOfFreedom());

    const double k_mlesac = params.getKmlesac ();
    switch (params.getScore()) {
        case ::vsac::ScoreMethod::SCORE_METHOD_RANSAC :
            quality = RansacQuality::create(points_val_size, threshold, error_val); break;
        case ::vsac::ScoreMethod::SCORE_METHOD_MSAC :
            quality = MsacQuality::create(points_val_size, threshold, error_val, k_mlesac); break;
        case ::vsac::ScoreMethod::SCORE_METHOD_MAGSAC :
            quality = MagsacQuality::create(max_thr, points_val_size, error_val, gamma_generator,
                threshold, params.getDegreesOfFreedom(),  params.getSigmaQuantile(),
                params.getUpperIncompleteOfSigmaQuantile(),
                params.getLowerIncompleteOfSigmaQuantile(), params.getC()); break;
        case ::vsac::ScoreMethod::SCORE_METHOD_LMEDS :
            quality = LMedsQuality::create(points_val_size, threshold, error_val); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Score is not imeplemeted!");
    }

    const auto is_svd_solver = params.getRansacSolver() == ::vsac::SVD_SOLVER;
    if (params.isHomography()) {
        degeneracy = HomographyDegeneracy::create(points_sample);
        if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
            min_solver = HomographySVDSolver::create(points_sample);
        else min_solver = HomographyMinimalSolver4ptsGEM::create(points_sample);
        non_min_solver = HomographyNonMinimalSolver::create(points_val); // use validation points for non-minimal solver
        estimator = HomographyEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params.isFundamental()) {
        degeneracy = FundamentalDegeneracy::create(state++, quality, points_sample, points_val, min_sample_size,
               params.getPlaneAndParallaxIters(), std::max(threshold, 8.) /*sqr homogr thr*/, inner_inlier_thr_sqr, K1, K2);
        if (K1.empty() || K2.empty())
            degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(
                params.getImage1Size().width/2., params.getImage1Size().height/2., params.getImage2Size().width/2., params.getImage2Size().height/2.);
        if(min_sample_size == 7) {
            if (is_svd_solver)
                min_solver = FundamentalSVDSolver::create(points_sample);
            else min_solver = FundamentalMinimalSolver7pts::create(points_sample);
        } else min_solver = FundamentalMinimalSolver8pts::create(points_sample);
        non_min_solver = EpipolarNonMinimalSolver::create(points_val, true);
        estimator = FundamentalEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params.isEssential()) {
        degeneracy = EssentialDegeneracy::create(points_sample, min_sample_size);
        min_solver = EssentialMinimalSolverStewenius5pts::create(points_sample, is_svd_solver);
        non_min_solver = EssentialNonMinimalSolverViaF::create(image_points_val, K1, K2);
        estimator = EssentialEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params.isPnP()) {
        degeneracy = makePtr<Degeneracy>();
        if (min_sample_size == 3) {
            min_solver = P3PSolver::create(points_sample, calib_points_sample, K1);
            non_min_solver = DLSPnP::create(points_val, calib_points_val, K1);
        } else {
            if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
                min_solver = PnPSVDSolver::create(points_sample);
            else min_solver = PnPMinimalSolver6Pts::create(points_sample);
            non_min_solver = PnPNonMinimalSolver::create(points_val);
        }
        estimator = PnPEstimator::create(min_solver, non_min_solver);
    } else if (params.getEstimator() == ::vsac::EstimationMethod::Affine) {
        degeneracy = makePtr<Degeneracy>();
        min_solver = AffineMinimalSolver::create(points_sample);
        non_min_solver = AffineNonMinimalSolver::create(points_val, cv::noArray(), cv::noArray());
        estimator = AffineEstimator::create(min_solver, non_min_solver);
    } else CV_Error(cv::Error::StsNotImplemented, "Estimator not implemented!");

    switch (params.getSampler()) {
        case ::vsac::SamplingMethod::SAMPLING_UNIFORM:
            if (params.isQuasiSampling())
                sampler = QuasiUniformSampler::create(state++, min_sample_size, points_sample_size);
            else sampler = UniformSampler::create(state++, min_sample_size, points_sample_size);
            break;
        case ::vsac::SamplingMethod::SAMPLING_PROSAC:
            if (!parallel_call) // for parallel only one PROSAC sampler
                sampler = ProsacSampler::create(state++, points_sample_size, min_sample_size, 200000);
            break;
        case ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC:
            sampler = ProgressiveNapsac::create(state++, points_sample_size, min_sample_size, layers, 20); break;
        case ::vsac::SamplingMethod::SAMPLING_NAPSAC:
            sampler = NapsacSampler::create(state++, points_sample_size, min_sample_size, graph); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Sampler is not implemented!");
    }

    const bool is_sprt = params.getVerifier() == ::vsac::VerificationMethod::SprtVerifier || params.getVerifier() == ::vsac::VerificationMethod::ASPRT;
    if (is_sprt)
        verifier = AdaptiveSPRT::create(state++, quality, points_val_size, params.getScore() == ::vsac::ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
            params.getSPRTepsilon(), params.getSPRTdelta(), params.getTimeForModelEstimation(),
            params.getSPRTavgNumModels(), params.getScore(), k_mlesac, params.getVerifier() == ::vsac::VerificationMethod::ASPRT);
    else if (params.getVerifier() == ::vsac::VerificationMethod::NullVerifier)
        verifier = ModelVerifier::create();
    else CV_Error(cv::Error::StsNotImplemented, "Verifier is not imeplemented!");

    // use sample points for termination
    if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROSAC) {
        termination = ProsacTerminationCriteria::create(parallel_call ? nullptr : sampler.dynamicCast<ProsacSampler>(), error_sample,
            points_sample_size, min_sample_size, params.getConfidence(), params.getMaxIters(), 100, 0.05, 0.05, threshold,
            parallel_call ? _termination.dynamicCast<ProsacTerminationCriteria>()->getNonRandomInliers() : std::vector<int>());
    } else if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
        if (is_sprt)
            termination = SPRTPNapsacTermination::create(((AdaptiveSPRT *)verifier.get())->getSPRTvector(),
                    params.getConfidence(), points_sample_size, min_sample_size,
                    params.getMaxIters(), params.getRelaxCoef());
        else
            termination = StandardTerminationCriteria::create (params.getConfidence(),
                   points_sample_size, min_sample_size, params.getMaxIters());
    } else if (is_sprt) {
        termination = SPRTTermination::create(((AdaptiveSPRT *) verifier.get())->getSPRTvector(),
             params.getConfidence(), points_sample_size, min_sample_size, params.getMaxIters());
    } else
        termination = StandardTerminationCriteria::create
            (params.getConfidence(), points_sample_size, min_sample_size, params.getMaxIters());

    if (! params.isParallel() || parallel_call) { // if normal ransac or parallel call
        if (params.getLO() != ::vsac::LocalOptimMethod::LOCAL_OPTIM_NULL) {
            lo_sampler = UniformRandomGenerator::create(state, points_val_size, params.getLOSampleSize());
            // do not use termination with two sets of points
            lo_termination = nullptr; //StandardTerminationCriteria::create(params.getConfidence(), points_val_size, min_sample_size, params.getMaxIters());
            switch (params.getLO()) {
                case ::vsac::LocalOptimMethod::LOCAL_OPTIM_INNER_LO:
                    lo = SimpleLocalOptimization::create(quality, estimator, lo_termination, lo_sampler,
                         params.getLOInnerMaxIters(), inner_inlier_thr_sqr); break;
                case ::vsac::LocalOptimMethod::LOCAL_OPTIM_INNER_AND_ITER_LO:
                    lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler,
                         points_val_size, threshold, true, params.getLOIterativeSampleSize(),
                         params.getLOInnerMaxIters(), params.getLOIterativeMaxIters(),
                         params.getLOThresholdMultiplier()); break;
                case ::vsac::LocalOptimMethod::LOCAL_OPTIM_GC:
                    lo = GraphCut::create(estimator, quality, graph, lo_sampler, threshold,
                       params.getGraphCutSpatialCoherenceTerm(), params.getLOInnerMaxIters(), lo_termination); break;
                case ::vsac::LocalOptimMethod::LOCAL_OPTIM_SIGMA:
                    lo = SigmaConsensus::create(estimator, quality, verifier, gamma_generator,
                         params.getLOSampleSize(), params.getLOInnerMaxIters(),
                         params.getDegreesOfFreedom(), params.getSigmaQuantile(),
                         params.getUpperIncompleteOfSigmaQuantile(), params.getC(), max_thr, lo_termination); break;
                default: CV_Error(cv::Error::StsNotImplemented , "Local Optimization is not implemented!");
            }
        }
    }
#ifdef DEBUG
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - init_time).count() << " initialize fnc\n";
#endif
}

int processPoints (const ::vsac::Params &params, cv::InputArray points1, cv::InputArray points2, const Mat &K1, const Mat &K2, InputArray dist_coeff1, InputArray dist_coeff2,
        Mat &points, Mat &calib_points) {
    int points_size;
    Mat undist_points1, undist_points2;
    if (params.isPnP()) {
        if (! K1.empty()) {
            if (! dist_coeff1.empty()) {
                // undistortPoints also calibrate points using K
                undistortPoints(points1.isContinuous() ? points1 : points1.getMat().clone(), undist_points1, K1, dist_coeff1);
                points_size = mergePoints(undist_points1, points2, points, true);
                Utils::normalizeAndDecalibPointsPnP (K1, points, calib_points);
            } else {
                points_size = mergePoints(points1, points2, points, true);
                Utils::calibrateAndNormalizePointsPnP(K1, points, calib_points);
            }
        } else points_size = mergePoints(points1, points2, points, true);
    } else {
        if (params.isEssential()) {
            if (! dist_coeff1.empty() || ! dist_coeff2.empty()) {
                // undistortPoints also calibrate points using K
                if (! dist_coeff1.empty()) undistortPoints(points1.isContinuous() ? points1 : points1.getMat().clone(), undist_points1, K1, dist_coeff1);
                else undist_points1 = points1.getMat();
                if (! dist_coeff2.empty()) undistortPoints(points2.isContinuous() ? points2 : points2.getMat().clone(), undist_points2, K2, dist_coeff2);
                else undist_points2 = points2.getMat();
                points_size = mergePoints(undist_points1, undist_points2, calib_points, false);
            } else {
                points_size = mergePoints(points1, points2, points, false);
                Utils::calibratePoints(K1, K2, points, calib_points);
            }
        } else {
            points_size = mergePoints(points1, points2, points, false);
        }
    }
    return points_size;
}

VSAC::VSAC (const ::vsac::Params &params_, cv::InputArray points1, cv::InputArray points2,
            cv::InputArray K1_, cv::InputArray K2_, cv::InputArray dist_coeff1, cv::InputArray dist_coeff2)
        : VSAC(params_, points1, points2, points1, points2, K1_, K2_, dist_coeff1, dist_coeff2) {}

VSAC::VSAC (const ::vsac::Params &params_, cv::InputArray points1_sample, cv::InputArray points2_sample,
            cv::InputArray points1_val, cv::InputArray points2_val,
            cv::InputArray K1_, cv::InputArray K2_, cv::InputArray dist_coeff1, cv::InputArray dist_coeff2)
            : params(params_) {
#ifdef DEBUG
    const auto init_time = std::chrono::steady_clock::now();
#endif
    _state = params.getRandomGeneratorState();
    threshold = params.getThreshold();
    max_thr = std::max(threshold, params.getMaximumThreshold());
    parallel = params.isParallel();
    if (params.isPnP()) {
        if (! K1_.empty()) {
            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
        } else if (params.getEstimator() == ::vsac::EstimationMethod::P3P) {
            std::cerr << "For P3P, intrinsic matrix must be provided!\n";
            exit(EXIT_FAILURE);
        }
    } else if (params.isEssential() || params.isFundamental()) {
        if (! K1_.empty() && ! K2_.empty()) {
            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
            K2 = K2_.getMat(); K2.convertTo(K2, CV_64F);
            if (params.isEssential()) {
                threshold = Utils::getCalibratedThreshold(threshold, K1, K2);
                max_thr = Utils::getCalibratedThreshold(max_thr, K1, K2);
            }
        } else if (params.isEssential()) {
            std::cerr << "For essential matrix estimation, both instrinsic matrices must be provided!\n";
            exit(EXIT_FAILURE);
        }
    }
    points_sample_size = processPoints(params, points1_sample, points2_sample, K1, K2, dist_coeff1, dist_coeff2, points_sample, calib_points_sample);
    points_val_size    = processPoints(params, points1_val   , points2_val   , K1, K2, dist_coeff1, dist_coeff2, points_val   , calib_points_val   );

    // Since error function output squared error distance, so make
    // threshold squared as well
    threshold *= threshold;

    // use sample points here
    if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_NAPSAC) {
        if (params.getNeighborsSearch() == ::vsac::NeighborSearchMethod::NEIGH_GRID) {
            graph = GridNeighborhoodGraph::create(points_sample, points_sample_size, params.getCellSize(), params.getCellSize(), params.getCellSize(), params.getCellSize(), 10);
        } else if (params.getNeighborsSearch() == ::vsac::NeighborSearchMethod::NEIGH_FLANN_KNN) {
            graph = FlannNeighborhoodGraph::create(points_sample, points_sample_size,params.getKNN(), false, 5, 1);
        } else if (params.getNeighborsSearch() == ::vsac::NeighborSearchMethod::NEIGH_FLANN_RADIUS) {
            assert(false && "check if it RadiusSearchNeighborhoodGraph works properly!");
            graph = RadiusSearchNeighborhoodGraph::create(points_sample, points_sample_size,params.getGraphRadius(), 5, 1);
        } else CV_Error(cv::Error::StsNotImplemented, "Graph type is not implemented!");
    }

    // use validation points here
    if (params.getLO() == ::vsac::LocalOptimMethod::LOCAL_OPTIM_GC) {
        if (params.getNeighborsSearch() == ::vsac::NeighborSearchMethod::NEIGH_GRID) {
            graph = GridNeighborhoodGraph::create(points_val, points_val_size, params.getCellSize(), params.getCellSize(), params.getCellSize(), params.getCellSize(), 10);
        } else if (params.getNeighborsSearch() == ::vsac::NeighborSearchMethod::NEIGH_FLANN_KNN) {
            graph = FlannNeighborhoodGraph::create(points_val, points_val_size,params.getKNN(), false, 5, 1);
        } else if (params.getNeighborsSearch() == ::vsac::NeighborSearchMethod::NEIGH_FLANN_RADIUS) {
            assert(false && "check if it RadiusSearchNeighborhoodGraph works properly!");
            graph = RadiusSearchNeighborhoodGraph::create(points_val, points_val_size,params.getGraphRadius(), 5, 1);
        } else CV_Error(cv::Error::StsNotImplemented, "Graph type is not implemented!");
    }

    if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
        CV_CheckEQ((int)params.isPnP(), 0, "ProgressiveNAPSAC for PnP is not implemented!");
        const auto &cell_number_per_layer = params.getGridCellNumber();
        layers.reserve(cell_number_per_layer.size());
        const auto * const pts = (float *) points_sample.data;
        float img1_width = 0, img1_height = 0, img2_width = 0, img2_height = 0;
        for (int i = 0; i < 4 * points_sample_size; i += 4) {
            if (pts[i    ] > img1_width ) img1_width  = pts[i    ];
            if (pts[i + 1] > img1_height) img1_height = pts[i + 1];
            if (pts[i + 2] > img2_width ) img2_width  = pts[i + 2];
            if (pts[i + 3] > img2_height) img2_height = pts[i + 3];
        }
        // Create grid graphs (overlapping layes of given cell numbers)
        for (int layer_idx = 0; layer_idx < (int)cell_number_per_layer.size(); layer_idx++) {
            const int cell_number = cell_number_per_layer[layer_idx];
            if (layer_idx > 0)
                if (cell_number_per_layer[layer_idx-1] <= cell_number)
                    CV_Error(cv::Error::StsError, "Progressive NAPSAC sampler: "
                                                  "Cell number in layers must be in decreasing order!");
            layers.emplace_back(GridNeighborhoodGraph::create(points_sample, points_sample_size,
        (int)(img1_width / (float)cell_number), (int)(img1_height / (float)cell_number),
        (int)(img2_width / (float)cell_number), (int)(img2_height / (float)cell_number), 10));
        }
    }

    // update points by calibrated for Essential matrix after graph is calculated
    if (params.isEssential()) {
        image_points_sample = points_sample;
        points_sample = calib_points_sample;
        image_points_val = points_val;
        points_val = calib_points_val;
        // if maximum calibrated threshold significanlty differs threshold then set upper bound
        if (max_thr > 10*threshold)
            max_thr = sqrt(10*threshold); // max thr will be squared after
    }

    /*if (params_.isHomography() || params_.isFundamental() || params.getEstimator() == ::vsac::EstimationMethod::Affine) {
        if (params_.getLO() != ::vsac::LOCAL_OPTIM_NULL && params.getFinalPolisher() != ::vsac::NonePolisher) {
            const auto normTr = NormTransform::create(points);
            std::vector<int> sample (points_size);
            for (int i = 0; i < points_size; i++) sample[i] = i;
                normTr->getNormTransformation(norm_points, sample, points_size, T1, T2);
        }
    }*/

    initialize (_state, _min_solver, _non_min_solver, _gamma_generator, _error_sample, _error_val, _estimator, _degeneracy, _quality,
            _model_verifier, _local_optimization, _termination, _sampler, _lo_sampler, _lo_termination, false/*parallel*/);
    quality_sample = RansacQuality::create(points_sample_size, threshold, _error_sample);

    switch (params.getFinalPolisher()) {
        case ::vsac::PolishingMethod::CovPolisher:
            if (params.isFundamental()) {
                cov_polisher = CovarianceEpipolarSolver::create(points_val, true);
            } else if (params.isHomography()) {
                cov_polisher = CovarianceHomographySolver::create(points_val);
            } else if (params.isEssential()) {
                cov_polisher = CovarianceEpipolarSolver::create(calib_points_val, false);
                if (! params.isEnforceRank())
                    cov_polisher->setEnforceRankConstraint(false);
            } else if (params.getEstimator() == ::vsac::EstimationMethod::Affine) {
                cov_polisher = CovarianceAffineSolver::create(points_val);
            } else assert(false && "covariance polisher not implemented\n");
            polisher = CovariancePolisher::create(_degeneracy, _quality, cov_polisher, params.getFinalLSQIterations());
            break;
        case ::vsac::PolishingMethod::MAGSAC:
            if (_gamma_generator == nullptr)
                _gamma_generator = GammaValues::create(params.getDegreesOfFreedom());
            polisher = WeightedPolisher::create(_degeneracy, _quality, _non_min_solver, _gamma_generator, 10,
                    params.getDegreesOfFreedom(), params.getUpperIncompleteOfSigmaQuantile(),params.getC(), max_thr);
            break;
        case ::vsac::PolishingMethod ::LSQPolisher:
            polisher = LeastSquaresPolishing::create(_estimator, _quality, params.getFinalLSQIterations());
            break;
        default : break;
    }
}
}}

namespace vsac {
    bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
           Output &output, cv::InputArray K1_, cv::InputArray K2_,
           cv::InputArray dist_coeff1, cv::InputArray dist_coeff2) {
        cv::vsac::VSAC vsac (params, points1, points2, K1_, K2_, dist_coeff1, dist_coeff2);
        return vsac.run(output);
    }
    bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
           cv::InputArray points1_val, cv::InputArray points2_val,
           Output &output, cv::InputArray K1_, cv::InputArray K2_,
           cv::InputArray dist_coeff1, cv::InputArray dist_coeff2) {
        cv::vsac::VSAC vsac (params, points1, points2, points1_val, points2_val, K1_, K2_, dist_coeff1, dist_coeff2);
        return vsac.run(output);
    }
}
