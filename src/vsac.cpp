//#define DEBUG true
#define DEBUG false

#include "precomp.hpp"
#include <atomic>

namespace cv { namespace vsac {
void initialize (int state, int points_size, double threshold, double max_thr, const ::vsac::Params &params, const Mat &points, 
        const Mat &calib_points, const Mat &image_points, const std::vector<Ptr<NeighborhoodGraph>> &layers, const std::vector<std::vector<int>> &close_pts_mask,
        const Mat &K1, const Mat &K2, const Ptr<NeighborhoodGraph> &graph, Ptr<MinimalSolver> &min_solver, 
        Ptr<NonMinimalSolver> &non_min_solver, Ptr<GammaValues> &gamma_generator, Ptr<Error> &error, Ptr<Estimator> &estimator, 
        Ptr<Degeneracy> &degeneracy, Ptr<Quality> &quality,
        Ptr<ModelVerifier> &verifier, Ptr<LocalOptimization> &lo, Ptr<Termination> &termination,
        Ptr<Sampler> &sampler, Ptr<RandomGenerator> &lo_sampler, bool parallel_call) {
#if DEBUG
    const auto init_time = std::chrono::steady_clock::now();
#endif
    const int min_sample_size = params.getSampleSize();

    // inner inlier threshold will be used in LO to obtain inliers
    // additionally in DEGENSAC for F
    double inner_inlier_thr_sqr = threshold;
    if (params.isHomography()) inner_inlier_thr_sqr = std::max(inner_inlier_thr_sqr, 5.25); // at least 2.5 px
    else if (params.isFundamental()) inner_inlier_thr_sqr = std::max(inner_inlier_thr_sqr, 4.); // at least 2 px

    switch (params.getError()) {
        case ::vsac::ErrorMetric::SYMM_REPR_ERR:
            error = ReprojectionErrorSymmetric::create(points); break;
        case ::vsac::ErrorMetric::FORW_REPR_ERR:
            if (params.getEstimator() == ::vsac::EstimationMethod::Affine)
                error = ReprojectionErrorAffine::create(points);
            else error = ReprojectionErrorForward::create(points);
            break;
        case ::vsac::ErrorMetric::SAMPSON_ERR:
            error = SampsonError::create(points); break;
        case ::vsac::ErrorMetric::SGD_ERR:
            error = SymmetricGeometricDistance::create(points); break;
        case ::vsac::ErrorMetric::RERPOJ:
            error = ReprojectionErrorPmatrix::create(points); break;
        default: CV_Error(cv::Error::StsNotImplemented , "Error metric is not implemented!");
    }

    if (params.getScore() == ::vsac::ScoreMethod::SCORE_METHOD_MAGSAC || params.getLO() == ::vsac::LocalOptimMethod::LOCAL_OPTIM_SIGMA)
        gamma_generator = GammaValues::create(params.getDegreesOfFreedom());

    const double k_mlesac = params.getKmlesac ();
    switch (params.getScore()) {
        case ::vsac::ScoreMethod::SCORE_METHOD_RANSAC :
            quality = RansacQuality::create(points_size, threshold, error); break;
        case ::vsac::ScoreMethod::SCORE_METHOD_MSAC :
            quality = MsacQuality::create(points_size, threshold, error, k_mlesac); break;
        case ::vsac::ScoreMethod::SCORE_METHOD_MAGSAC :
            quality = MagsacQuality::create(max_thr, points_size, error, gamma_generator,
                threshold, params.getDegreesOfFreedom(),  params.getSigmaQuantile(),
                params.getUpperIncompleteOfSigmaQuantile(),
                params.getLowerIncompleteOfSigmaQuantile(), params.getC()); break;
        case ::vsac::ScoreMethod::SCORE_METHOD_LMEDS :
            quality = LMedsQuality::create(points_size, threshold, error); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Score is not imeplemeted!");
    }

    if (params.isHomography()) {
        degeneracy = HomographyDegeneracy::create(points);
        if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
            min_solver = HomographySVDSolver::create(points);
        else min_solver = HomographyMinimalSolver4ptsGEM::create(points);
        non_min_solver = HomographyNonMinimalSolver::create(points);
        estimator = HomographyEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params.isFundamental()) {
        degeneracy = FundamentalDegeneracy::create(state++, quality, points, min_sample_size,
               params.getPlaneAndParallaxIters(), 8. /*sqr homogr thr*/, inner_inlier_thr_sqr, K1, K2);
        degeneracy->setClosePointsMask(close_pts_mask);
        auto img1_size = params.getImage1Size(), img2_size = params.getImage2Size();
        // std::cout << K1.empty() << " " << img1_size.width << " " << img1_size.height <<"\n";
        if (K1.empty()) {
            if (img1_size.width != 0 && img1_size.height != 0) {
                if (img1_size.width < img1_size.height) std::swap(img1_size.width, img1_size.height);
                if (img2_size.height != 0 && img2_size.width != 0) {
                    if (img2_size.width < img2_size.height) std::swap(img2_size.width, img2_size.height);
                    degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(img1_size.width/2., img1_size.height/2., img2_size.width/2., img2_size.height/2.);
                } else degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(img1_size.width/2., img1_size.height/2.);
            }
        }

        if (K1.empty() && img1_size.width != 0 && img1_size.height != 0) {
            if (img1_size.width > img1_size.height)
                 degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(img1_size.width/2., img1_size.height/2.);
            else degeneracy.dynamicCast<FundamentalDegeneracy>()->setPrincipalPoint(img1_size.height/2., img1_size.width/2.);
        }
        if(min_sample_size == 7) {
            if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
                min_solver = FundamentalSVDSolver::create(points);
            else min_solver = FundamentalMinimalSolver7pts::create(points);
        } else min_solver = FundamentalMinimalSolver8pts::create(points);
        non_min_solver = EpipolarNonMinimalSolver::create(points, true);
        estimator = FundamentalEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params.isEssential()) {
        degeneracy = EssentialDegeneracy::create(points, min_sample_size);
        min_solver = EssentialMinimalSolverStewenius5pts::create(points, params.getRansacSolver() == ::vsac::SVD_SOLVER);
        non_min_solver = EpipolarNonMinimalSolver::create(points, false);
        estimator = EssentialEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params.isPnP()) {
        degeneracy = makePtr<Degeneracy>();
        if (min_sample_size == 3) {
            min_solver = P3PSolver::create(points, calib_points, K1);
            non_min_solver = DLSPnP::create(points, calib_points, K1);
        } else {
            if (params.getRansacSolver() == ::vsac::SVD_SOLVER)
                min_solver = PnPSVDSolver::create(points);
            else min_solver = PnPMinimalSolver6Pts::create(points);
            non_min_solver = PnPNonMinimalSolver::create(points);
        }
        estimator = PnPEstimator::create(min_solver, non_min_solver);
    } else if (params.getEstimator() == ::vsac::EstimationMethod::Affine) {
        degeneracy = makePtr<Degeneracy>();
        min_solver = AffineMinimalSolver::create(points);
        non_min_solver = AffineNonMinimalSolver::create(points);
        estimator = AffineEstimator::create(min_solver, non_min_solver);
    } else CV_Error(cv::Error::StsNotImplemented, "Estimator not implemented!");

    switch (params.getSampler()) {
        case ::vsac::SamplingMethod::SAMPLING_UNIFORM:
            if (params.isQuasiSampling())
                sampler = QuasiUniformSampler::create(state++, min_sample_size, points_size);
            else sampler = UniformSampler::create(state++, min_sample_size, points_size);
            break;
        case ::vsac::SamplingMethod::SAMPLING_PROSAC:
            if (!parallel_call) // for parallel only one PROSAC sampler
                sampler = ProsacSampler::create(state++, points_size, min_sample_size, 200000);
            break;
        case ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC:
            sampler = ProgressiveNapsac::create(state++, points_size, min_sample_size, layers, 20); break;
        case ::vsac::SamplingMethod::SAMPLING_NAPSAC:
            sampler = NapsacSampler::create(state++, points_size, min_sample_size, graph); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Sampler is not implemented!");
    }

    switch (params.getVerifier()) {
        case ::vsac::VerificationMethod::NullVerifier: verifier = ModelVerifier::create(); break;
        case ::vsac::VerificationMethod::SprtVerifier:
            verifier = AdaptiveSPRT::create(state++, error, quality, points_size, params.getScore() == ::vsac::ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
             params.getSPRTepsilon(), params.getSPRTdelta(), params.getTimeForModelEstimation(),
             params.getSPRTavgNumModels(), params.getScore(), params.isParallel() ? 0 : params.getMaxNumHypothesisToTestBeforeRejection(), k_mlesac, false); break;
        case ::vsac::VerificationMethod::ASPRT:
            verifier = AdaptiveSPRT::create(state++, error, quality, points_size, params.getScore() == ::vsac::ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
             params.getSPRTepsilon(), params.getSPRTdelta(), params.getTimeForModelEstimation(),
             params.getSPRTavgNumModels(), params.getScore(), params.isParallel() ? 0 : params.getMaxNumHypothesisToTestBeforeRejection(), k_mlesac); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Verifier is not imeplemented!");
    }

    const bool is_sprt = params.getVerifier() == ::vsac::VerificationMethod::SprtVerifier || params.getVerifier() == ::vsac::VerificationMethod::ASPRT;
    if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROSAC) {
        if (!parallel_call)
            termination = ProsacTerminationCriteria::create(sampler.dynamicCast<ProsacSampler>(), error,
                points_size, min_sample_size, params.getConfidence(), params.getMaxIters(), 100, 0.05, 0.05, threshold);
    } else if (params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
        if (is_sprt)
            termination = SPRTPNapsacTermination::create(((AdaptiveSPRT *)verifier.get())->getSPRTvector(),
                    params.getConfidence(), points_size, min_sample_size,
                    params.getMaxIters(), params.getRelaxCoef());
        else
            termination = StandardTerminationCriteria::create (params.getConfidence(),
                    points_size, min_sample_size, params.getMaxIters());
    } else if (is_sprt) {
        termination = SPRTTermination::create(((AdaptiveSPRT *) verifier.get())->getSPRTvector(),
             params.getConfidence(), points_size, min_sample_size, params.getMaxIters());
    } else
        termination = StandardTerminationCriteria::create
            (params.getConfidence(), points_size, min_sample_size, params.getMaxIters());

    if (params.getLO() != ::vsac::LocalOptimMethod::LOCAL_OPTIM_NULL) {
        lo_sampler = UniformRandomGenerator::create(state, points_size, params.getLOSampleSize());
        const auto lo_termination = StandardTerminationCriteria::create(params.getConfidence(), points_size, min_sample_size, params.getMaxIters());
        switch (params.getLO()) {
            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_INNER_LO:
                lo = SimpleLocalOptimization::create(degeneracy, quality, estimator, lo_termination, lo_sampler,
                     params.getLOInnerMaxIters(), inner_inlier_thr_sqr); break;
            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_INNER_AND_ITER_LO:
                lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler,
                     points_size, threshold, true, params.getLOIterativeSampleSize(),
                     params.getLOInnerMaxIters(), params.getLOIterativeMaxIters(),
                     params.getLOThresholdMultiplier()); break;
            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_GC:
                lo = GraphCut::create(estimator, error, quality, graph, lo_sampler, threshold,
                   params.getGraphCutSpatialCoherenceTerm(), params.getLOInnerMaxIters(), lo_termination); break;
            case ::vsac::LocalOptimMethod::LOCAL_OPTIM_SIGMA:
                lo = SigmaConsensus::create(estimator, error, quality, verifier, gamma_generator,
                     params.getLOSampleSize(), params.getLOInnerMaxIters(),
                     params.getDegreesOfFreedom(), params.getSigmaQuantile(),
                     params.getUpperIncompleteOfSigmaQuantile(), params.getC(), max_thr, lo_termination); break;
            default: CV_Error(cv::Error::StsNotImplemented , "Local Optimization is not implemented!");
        }
    }
#if DEBUG
    std::cout << "Init time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - init_time).count() << '\n';
#endif
}

class VSAC {
protected:
    const Mat &points;
    const ::vsac::Params &params;
    const Ptr<Estimator> _estimator;
    const Ptr<Quality> _quality;
    const Ptr<Sampler> _sampler;
    const Ptr<Termination> _termination;
    const Ptr<ModelVerifier> _model_verifier;
    const Ptr<Degeneracy> _degeneracy;
    const Ptr<LocalOptimization> _local_optimization;
    const Ptr<FinalModelPolisher> model_polisher;

    const int points_size, state;
    const bool parallel;

    // parallel RANSAC data
    double threshold, max_thr;
    Mat K1, K2, calib_points, image_points;
    Ptr<NeighborhoodGraph> graph;
    std::vector<Ptr<NeighborhoodGraph>> layers;
    std::vector<std::vector<int>> close_pts_mask;
public:
    void setDataForParallel(double threshold_, double max_thr_, const Mat &K1_, const Mat &K2_, const Mat &calib_points_, const Mat &image_points_,
        const Ptr<NeighborhoodGraph> &graph_, const std::vector<Ptr<NeighborhoodGraph>> &layers_, const std::vector<std::vector<int>> &close_pts_mask_) {
        threshold = threshold_; max_thr = max_thr_; K1 = K1_; K2 = K2_; calib_points = calib_points_; image_points =  image_points_;
        graph = graph_; layers = layers_; close_pts_mask = close_pts_mask_;
    }

    VSAC (const Mat &points_, const ::vsac::Params &params_, int points_size_, const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
            const Ptr<Sampler> &sampler_, const Ptr<Termination> &termination_,
            const Ptr<ModelVerifier> &model_verifier_, const Ptr<Degeneracy> &degeneracy_,
            const Ptr<LocalOptimization> &local_optimization_, const Ptr<FinalModelPolisher> &model_polisher_,
            bool parallel_=false, int state_ = 0) :
            points (points_),
            params (params_), _estimator (estimator_), _quality (quality_), _sampler (sampler_),
            _termination (termination_), _model_verifier (model_verifier_),
            _degeneracy (degeneracy_), _local_optimization (local_optimization_),
            model_polisher (model_polisher_), points_size (points_size_), state(state_),
            parallel(parallel_) {}

    bool run(::vsac::Output &ransac_output) {
        if (points_size < params.getSampleSize())
            return false;
#if DEBUG
        const auto begin_time = std::chrono::steady_clock::now();
#endif
        const bool LO = params.getLO() != ::vsac::LocalOptimMethod::LOCAL_OPTIM_NULL, IS_QUASI_SAMPLING = params.isQuasiSampling();
        Score best_score;
        Mat best_model;
        const int MAX_MODELS_ADAPT = 21, MAX_ITERS_ADAPT = MAX_MODELS_ADAPT/*assume at least 1 model from 1 sample, todo: think about extreme cases*/, sample_size = params.getSampleSize();
        const double IOU_SIMILARITY_THR = 0.80;
        std::vector<int> non_degen_sample, best_sample;

        double lambda_non_random = 0;
        int final_iters, num_lo_runs = 0, num_total_tested_models = 0, num_so_far_the_best = 0;

#if DEBUG
        double est_time = 0, eval_time = 0, polisher_time = 0, lo_time = 0;
        int num_degenerate_cases = 0, num_degensac_runs = 0;
        double degensac_time = 0;
#endif

        // non-random
        const int MAX_TEST_MODELS_NONRAND = params.isNonRandomnessTest() ? MAX_MODELS_ADAPT : 0;
        std::vector<Mat> models_for_random_test; models_for_random_test.reserve(MAX_TEST_MODELS_NONRAND);
        std::vector<std::vector<int>> samples_for_random_test; samples_for_random_test.reserve(MAX_TEST_MODELS_NONRAND);

        // for the verification of F in the end
        bool last_model_from_LO = false, last_model_from_min_sample = true;
        cv::Mat best_model_not_from_LO;
        Score best_score_model_not_from_LO;
        /////////////////////////
        std::vector<bool> best_inliers_mask(points_size);
        if (!params.isEnforceRank())
            _estimator->enforceRankConstraint(false);
        if (! parallel) {
            // adaptive sprt test
            double IoU = 0, mean_time_estimation = 0, mean_num_est_models = 0, mean_time_evaluation = 0;
            bool adapt = params.isNonRandomnessTest() || params.getVerifier() == ::vsac::VerificationMethod ::ASPRT, was_LO_run = false;
            int min_non_random_inliers = 0, num_correspondences_of_bad_models = 0, iters = 0, max_iters = params.getMaxIters(),
                    correction_inls = 0, correction_models = 0, num_estimations = 0, non_planar_support = 0;
            Mat non_degenerate_model, lo_model;
            Score current_score, lo_score, non_degenerate_model_score;
            std::vector<bool> model_inliers_mask (points_size);
            std::vector<Mat> models(_estimator->getMaxNumSolutions());
            std::vector<int> sample(_estimator->getMinimalSampleSize()), supports(3*MAX_MODELS_ADAPT, 0);

            auto update_best = [&] (const Mat &new_model, const Score &new_score, bool last_model_lo=false) {
                _quality->getInliers(new_model, model_inliers_mask);
                if (!adapt && IS_QUASI_SAMPLING && new_score.inlier_number > 100) {// update quasi sampler
//                    std::cout << "update quasi sampler at iter " << iters << " with inliers " << new_score.inlier_number << '\n';
                    _sampler->updateSampler(model_inliers_mask);
                }
                // IoU is used for LO and adaption
                IoU = Utils::intersectionOverUnion(best_inliers_mask, model_inliers_mask);
#if DEBUG
//                if (last_model_lo)
                  std::cout << "UPDATE BEST, iters " << iters << " (" << best_score.score << "," << best_score.inlier_number << ") -> (" << new_score.score << ", " << new_score.inlier_number << ") IoU " << IoU << " from LO " << last_model_lo << '\n';
#endif
                if (!best_model.empty() && models_for_random_test.size() < MAX_TEST_MODELS_NONRAND && IoU < IOU_SIMILARITY_THR) {
                    // save old best model for non-randomness test if necessary
                    models_for_random_test.emplace_back(best_model.clone());
                    samples_for_random_test.emplace_back(best_sample);
                }
                if (!adapt) {
                    // update quality and verifier to save evaluation time of a model
                    _quality->setBestScore(new_score.score);
                    _model_verifier->update(new_score.inlier_number);
                } else {
                    if (IoU >= IOU_SIMILARITY_THR) {
                        correction_models++; correction_inls += best_score.inlier_number; // add second best
                    } else {
                        correction_models = 0; correction_inls = 0;
                    }
                }
                // update score, model, inliers and max iterations
                best_inliers_mask = model_inliers_mask;
                best_score = new_score;
                new_model.copyTo(best_model);
#if DEBUG
                num_so_far_the_best++;
#endif
                best_sample = sample;
                max_iters = _termination->update(best_model, best_score.inlier_number);
                last_model_from_LO = last_model_lo;
                if (!last_model_from_LO) {
                    best_model.copyTo(best_model_not_from_LO);
                    best_score_model_not_from_LO = best_score;
                }
            };

            auto runLO = [&] (int current_ransac_iters) {
                // std::cout << "run lo at iter " << current_ransac_iters << "\n";
                // if (_degeneracy->isDecisionable()) {
                    was_LO_run = true;
                    _local_optimization->setCurrentRANSACiter(current_ransac_iters);
                    // std::cout << "best score " << best_score.score << " ";
#if DEBUG
                    num_lo_runs++;
                    const auto temp_time = std::chrono::steady_clock::now();
#endif
                    if (_local_optimization->refineModel
                            (best_model, best_score, lo_model, lo_score) && lo_score.isBetter(best_score)){
#if DEBUG
                        lo_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                        std::cout << "START LO at " << iters << " BEST (" << best_score.score << "," << best_score.inlier_number << "), LO is BETTER (" << lo_score.score << ", " << lo_score.inlier_number << ")\n";
#endif
                        update_best(lo_model, lo_score, true);
                    }
#if DEBUG
                    else {
                        lo_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                        std::cout << "START LO at " << iters << " BEST (" << best_score.score << "," << best_score.inlier_number << "), LO is WORSE (" << lo_score.score << ", " << lo_score.inlier_number << ")\n";
                    }
#endif
                // }
            };
            for (; iters < max_iters; iters++) {
                _sampler->generateSample(sample);
                int number_of_models;
                if (adapt) {
                    const auto time_estimation = std::chrono::steady_clock::now();
                    number_of_models = _estimator->estimateModels(sample, models);
#if DEBUG
                    est_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_estimation).count();
#endif
                    if (iters != 0)
                        mean_time_estimation += std::chrono::duration_cast<std::chrono::microseconds>
                            (std::chrono::steady_clock::now() - time_estimation).count();
                    mean_num_est_models += number_of_models;
                    num_estimations++;
                } else {
#if DEBUG
                    const auto temp_time = std::chrono::steady_clock::now();
#endif
                    number_of_models = _estimator->estimateModels(sample, models);
#if DEBUG
                    est_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
#endif
                }
                for (int i = 0; i < number_of_models; i++) {
                    num_total_tested_models++;
                    if (adapt) {
                        const auto time_evaluation = std::chrono::steady_clock::now();
                        current_score = _quality->getScore(models[i]);
                        // std::cout << "iters " << iters << " model " << i << " score " << current_score.score << "\n";
                        if (iters != 0)
                            mean_time_evaluation += std::chrono::duration_cast<std::chrono::microseconds>
                                (std::chrono::steady_clock::now() - time_evaluation).count();
#if DEBUG
                        eval_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_evaluation).count();
#endif
                        num_correspondences_of_bad_models += current_score.inlier_number;
                        if (num_total_tested_models-1 < supports.size())
                            supports[num_total_tested_models-1] = current_score.inlier_number;
                    } else {
#if DEBUG
                        const auto temp_time = std::chrono::steady_clock::now();
#endif
                        if (_model_verifier->isModelGood(models[i])) {
                            if (!_model_verifier->getScore(current_score))
                                current_score = _quality->getScore(models[i]);
                        } else {
#if DEBUG
                            eval_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                            const auto sc = _quality->getScore(models[i]);
                            if (sc.isBetter(best_score))
                                std::cout << "SPRT REJECTED BETTER MODEL (" << sc.score << ", " << sc.inlier_number << ")\n";
#endif
                            continue;
                        }
#if DEBUG
                        eval_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
#endif
                    }
#if DEBUG
                    std::cout << "iter " << iters << " score (" << current_score.score << ", " << current_score.inlier_number << ")\n";
#endif
                    if (current_score.isBetter(best_score)) {
#if DEBUG
                        const auto temp_time = std::chrono::steady_clock::now();                        
                        const bool is_degen = _degeneracy->recoverIfDegenerate(sample, models[i], current_score, non_degenerate_model, non_degenerate_model_score, non_planar_support);
                        degensac_time += std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now() - temp_time).count();
                        num_degensac_runs++;
                        if (is_degen) {
                            num_degenerate_cases++;
#else
                        if (_degeneracy->recoverIfDegenerate(sample, models[i], current_score,
                                   non_degenerate_model, non_degenerate_model_score, non_planar_support)) {
#endif
#if DEBUG
                            std::cout << "IS DEGENERATE, new score (" << non_degenerate_model_score.score << ", " << non_degenerate_model_score.inlier_number << ")\n"; 
#endif
                            // check if best non degenerate model is better than so far the best model
                            if (non_degenerate_model_score.isBetter(best_score))
                                update_best(non_degenerate_model, non_degenerate_model_score);
                            else continue;
                        } else update_best(models[i], current_score);
                        if (iters < max_iters && LO && best_score.inlier_number > min_non_random_inliers && IoU < IOU_SIMILARITY_THR && !adapt)
                            runLO(iters); // update model by Local optimization
                    } // end of if so far the best score
                    else if (models_for_random_test.size() < MAX_TEST_MODELS_NONRAND) {
                        models_for_random_test.emplace_back(models[i].clone());
                        samples_for_random_test.emplace_back(sample);
                    }
                    if (iters > max_iters)
                        break; // break loop over models
                } // end loop of number of models
                if (adapt && iters >= MAX_ITERS_ADAPT && num_total_tested_models >= MAX_MODELS_ADAPT &&
                        num_correspondences_of_bad_models - best_score.inlier_number - correction_inls > 0) {
                    adapt = false;
                    ////////////////////////// update ///////////////////////
                    supports.resize(std::min(num_total_tested_models, (int)supports.size()));
                    std::sort(supports.begin(), supports.end());
                    double delta = (double)supports[supports.size()/2] / points_size;
                    const double cdf_99 = delta * points_size + 2.32*sqrt(delta * points_size * (1 - delta));
                    delta = 0;
                    int num_lower = 0;
                    for (num_lower = 0; num_lower < (int)supports.size(); num_lower++) {
                        if (supports[num_lower] < cdf_99) delta += supports[num_lower];
                        else break;
                    }
                    delta /= (num_lower * points_size);
                    if (std::isnan(delta)) delta = params.getSPRTdelta();
//                    std::cout << "delta avg " << delta_avg << " lambda avg " << delta_avg * points_size << " new delta " << delta << " lambda " << delta * points_size << "\n";
                    //////////////////////////////////////////////////////////////////////

//                    // https://keisan.casio.com/exec/system/14060745333941
                    min_non_random_inliers = (int)ceil(delta * points_size + 3.719*sqrt(delta * points_size * (1 - delta)));
#if DEBUG
                    std::cout << "ADAPT ENDED, iters " << iters << " models " << num_total_tested_models << " min non-rand inls " << min_non_random_inliers << ", delta " << delta << " mu (lambda) " << delta * points_size << " correction (inls) " << correction_inls << ", models " << correction_models << "\n";
#endif
                    _model_verifier->updateSPRT(mean_time_estimation / (num_estimations-1), mean_time_evaluation / ((num_total_tested_models-1) * points_size),
                            mean_num_est_models / num_estimations, delta, std::max((double)min_non_random_inliers/points_size, (double)best_score.inlier_number / points_size), best_score);
                }
                if (!adapt && LO && !was_LO_run && iters < max_iters && !best_model.empty() && best_score.inlier_number > min_non_random_inliers)
                    runLO(iters);
            } // end main while loop

            final_iters = iters;
            if (! was_LO_run && !best_model.empty() && LO)
                runLO(-1 /*use full iterations of LO*/);
        } else {
            const int MAX_THREADS = getNumThreads();
            const bool is_prosac = params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROSAC;

            std::atomic_bool success(false);
            std::atomic_int num_hypothesis_tested(0);
            std::atomic_int thread_cnt(0);
            std::atomic_int max_number_inliers(0);
            std::atomic<double> best_score_all(std::numeric_limits<double>::max());
            std::vector<Score> best_scores(MAX_THREADS);
            std::vector<Mat> best_models(MAX_THREADS);
            std::vector<int> num_tested_models_threads(MAX_THREADS);
            std::vector<std::vector<Mat>> tested_models_threads(MAX_THREADS);
            std::vector<std::vector<std::vector<int>>> tested_samples_threads(MAX_THREADS);
            std::vector<std::vector<int>> best_samples_threads(MAX_THREADS);

            std::vector<int> growth_function, non_random_inliers;
            const int min_termination_length = is_prosac ? _termination.dynamicCast<ProsacTerminationCriteria>()->getMinTerminationLength() : 0;
            const int growth_max_samples = 200000;
            std::atomic_int subset_size, termination_length;
            const double log_confidence = is_prosac ? log(1-params.getConfidence()) : 0;
            if (is_prosac) {
                non_random_inliers = _termination.dynamicCast<ProsacTerminationCriteria>()->getNonRandomInliers();
                growth_function = _sampler.dynamicCast<ProsacSampler>()->getGrowthFunction();
                subset_size = 2*sample_size; // n,  size of the current sampling pool
                termination_length = points_size;
            }
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            parallel_for_(Range(0, MAX_THREADS), [&](const Range & /*range*/) {
            if (!success) { // cover all if not success to avoid thread creating new variables
                const int thread_rng_id = thread_cnt++;
                int thread_state = state + thread_rng_id;
                bool adapt = params.getVerifier() == ::vsac::VerificationMethod ::ASPRT;
                int min_non_random_inliers = 0, num_correspondences_of_bad_models = 0, num_tested_models = 0,
                    num_estimations = 0, mean_num_est_models = 0;
                std::vector<Mat> tested_models_thread; tested_models_thread.reserve(MAX_TEST_MODELS_NONRAND);
                std::vector<std::vector<int>> tested_samples_thread; tested_samples_thread.reserve(MAX_TEST_MODELS_NONRAND);
                Ptr<UniformRandomGenerator> random_gen;
                if (is_prosac) random_gen = UniformRandomGenerator::create(thread_state);
                Ptr<Error> error;
                Ptr<Estimator> estimator;
                Ptr<Degeneracy> degeneracy;
                Ptr<Quality> quality;
                Ptr<ModelVerifier> model_verifier;
                Ptr<Sampler> sampler;
                Ptr<RandomGenerator> lo_sampler;
                Ptr<Termination> termination;
                Ptr<LocalOptimization> local_optimization;
                Ptr<FinalModelPolisher> polisher;
                Ptr<MinimalSolver> min_solver;
                Ptr<NonMinimalSolver> non_min_solver;
                Ptr<GammaValues> gamma_generator;

                initialize (thread_state, points_size, threshold, max_thr, params, points, calib_points, image_points, layers, close_pts_mask, K1, K2, graph, min_solver,
                    non_min_solver, gamma_generator, error, estimator, degeneracy, quality, model_verifier, local_optimization, termination, sampler, lo_sampler, true);

                double IoU = 0;
                Mat best_model_thread, non_degenerate_model, lo_model;
                Score best_score_thread, current_score, non_denegenerate_model_score, lo_score,best_score_all_threads;
                std::vector<int> sample(estimator->getMinimalSampleSize()), best_sample_thread, supports(3*MAX_MODELS_ADAPT, 0);
                std::vector<bool> best_inliers_mask(points_size, false), model_inliers_mask(points_size, false);
                std::vector<Mat> models(estimator->getMaxNumSolutions());
                int iters, max_iters = params.getMaxIters(), non_planar_support, correction_inls = 0, correction_models = 0, max_non_random_inliers = 0;
                auto update_best = [&] (const Score &new_score, const Mat &new_model) {
                    if (max_number_inliers < new_score.inlier_number)
                        max_number_inliers = new_score.inlier_number;
                    if (best_score_all > new_score.score)
                        best_score_all = new_score.score;
                    quality->getInliers(new_model, model_inliers_mask);
                    if (IS_QUASI_SAMPLING)
                        sampler->updateSampler(model_inliers_mask);
                    IoU = Utils::intersectionOverUnion(best_inliers_mask, model_inliers_mask);
                    if (!best_model_thread.empty() && tested_models_thread.size() < MAX_TEST_MODELS_NONRAND && IoU < IOU_SIMILARITY_THR) {
                        tested_models_thread.emplace_back(best_model_thread.clone());
                        tested_samples_thread.emplace_back(best_sample_thread);
                    }
                    if (! adapt) {
                        if (IoU >= IOU_SIMILARITY_THR) {
                            correction_models++; correction_inls += best_score.inlier_number; // add second best
                        } else {
                            correction_models = 0; correction_inls = 0;
                        }
                    }
                    best_score_all_threads = Score(max_number_inliers, best_score_all);
                    // copy new score to best score
                    best_score_thread = new_score;
                    best_sample_thread = sample;
                    best_inliers_mask = model_inliers_mask;
                    // remember best model
                    new_model.copyTo(best_model_thread);

                    // update upper bound of iterations
                    if (is_prosac) {
                        int predicted_iterations = max_iters;
                        /*
                         * The termination length n* is chosen to minimize k_n*(η0) subject to I_n* ≥ I_min n*;
                         * k_n*(η0) >= log(η0) / log(1 - (I_n* / n*)^m)
                         * g(k) <= n, I_n is number of inliers under termination length n.
                         */
                        const auto &errors = error->getErrors(best_model_thread);

                        // find number of inliers under g(k)
                        int num_inliers_under_termination_len = 0;
                        for (int pt = 0; pt < min_termination_length; pt++)
                            if (errors[pt] < threshold)
                                num_inliers_under_termination_len++;
                        int new_termination_length = points_size;
                        for (int termination_len = min_termination_length; termination_len < points_size;termination_len++){
                            if (errors[termination_len /* = point*/] < threshold) {
                                num_inliers_under_termination_len++;
                                // non-random constraint must satisfy I_n* ≥ I_min n*.
                                if (num_inliers_under_termination_len < non_random_inliers[termination_len])
                                    continue;
                                // add 1 to termination length since num_inliers_under_termination_len is updated
                                const double new_max_samples = log_confidence / log(1 -
                                        std::pow(static_cast<double>(num_inliers_under_termination_len)
                                        / (termination_len+1), sample_size));

                                if (! std::isinf(new_max_samples) && predicted_iterations > new_max_samples) {
                                    predicted_iterations = static_cast<int>(new_max_samples);
                                    if (predicted_iterations == 0) break;
                                    new_termination_length = termination_len;
                                }
                            }
                        }
                        if (predicted_iterations < max_iters)
                            max_iters = predicted_iterations;

                        // compare also when termination length = points_size,
                        // so inliers under termination length is total number of inliers:
                        const double predicted_iters = log_confidence / log(1 - std::pow
                                (static_cast<double>(best_score_thread.inlier_number) / points_size, sample_size));

                        if (! std::isinf(predicted_iters) && predicted_iters < max_iters)
                            max_iters = static_cast<int>(predicted_iters)+1;

                        // update termination length
                        if (new_termination_length < termination_length)
                            termination_length = new_termination_length;
                    } else max_iters = termination->update(best_model_thread, max_number_inliers);
                };
                bool was_LO_run = false;
                auto runLO = [&] (int current_ransac_iters) {
                    was_LO_run = true;
                    local_optimization->setCurrentRANSACiter(current_ransac_iters);
                    if (local_optimization->refineModel
                            (best_model_thread, best_score_thread, lo_model, lo_score) && lo_score.isBetter(best_score_thread)){
                        update_best(lo_score, lo_model);
                    }
                };
                for (iters = 0; iters < max_iters && !success; iters++) {
                    success = num_hypothesis_tested++ > max_iters;
                    if (iters % 10) {
                        if (!adapt) {
                            // Synchronize threads. just to speed verification of model.
                            quality->setBestScore(std::min(best_score_thread.score, (double)best_score_all));
                            model_verifier->update(std::max(best_score.inlier_number, (int)max_number_inliers));
                        }
                    }

                    if (is_prosac) {
                        if (num_hypothesis_tested > growth_max_samples) {
                            // if PROSAC has not converged to solution then do uniform sampling.
                            random_gen->generateUniqueRandomSet(sample, sample_size, points_size);
                        } else {
                            if (num_hypothesis_tested >= growth_function[subset_size-1] && subset_size < termination_length-MAX_THREADS) {
                                subset_size++;
                                if (subset_size >= points_size) subset_size = points_size-1;
                            }
                            if (growth_function[subset_size-1] < num_hypothesis_tested) {
                                // The sample contains m-1 points selected from U_(n-1) at random and u_n
                                random_gen->generateUniqueRandomSet(sample, sample_size-1, subset_size-1);
                                sample[sample_size-1] = subset_size-1;
                            } else
                                // Select m points from U_n at random.
                                random_gen->generateUniqueRandomSet(sample, sample_size, subset_size);
                        }
                    } else sampler->generateSample(sample); // use local sampler

                    const int number_of_models = estimator->estimateModels(sample, models);
                    if (adapt) {
                        num_estimations++;
                        mean_num_est_models += number_of_models;
                    }
                    for (int i = 0; i < number_of_models; i++) {
                        num_tested_models++;
                        if (adapt) {
                            current_score = quality->getScore(models[i]);
                            const int non_rand_m = current_score.inlier_number;
                            num_correspondences_of_bad_models += non_rand_m;
                            if (num_tested_models-1 < supports.size())
                                supports[num_tested_models-1] = current_score.inlier_number;
                        } else {
                            if (model_verifier->isModelGood(models[i])) {
                                if (!model_verifier->getScore(current_score))
                                    current_score = quality->getScore(models[i]);
                            } else continue;
                        }

                        if (current_score.isBetter(best_score_all_threads)) {
                            if (degeneracy->recoverIfDegenerate(sample, models[i], current_score,
                                    non_degenerate_model, non_denegenerate_model_score, non_planar_support)) {
                                // check if best non degenerate model is better than so far the best model
                                if (non_denegenerate_model_score.isBetter(best_score_thread))
                                    update_best(non_denegenerate_model_score, non_degenerate_model);
                                else continue;
                            } else update_best(current_score, models[i]);

                            if (!adapt && LO && num_hypothesis_tested < max_iters && IoU < IOU_SIMILARITY_THR &&
                                    best_score_thread.inlier_number > min_non_random_inliers)
                                runLO(iters);
                        } // end of if so far the best score
                        else if (tested_models_thread.size() < MAX_TEST_MODELS_NONRAND) {
                            tested_models_thread.emplace_back(models[i].clone());
                            tested_samples_thread.emplace_back(sample);
                        }
                        if (num_hypothesis_tested > max_iters) {
                            success = true; break;
                        }
                    } // end loop of number of models
                    if (adapt && iters >= MAX_ITERS_ADAPT && num_tested_models >= MAX_MODELS_ADAPT &&
                            num_correspondences_of_bad_models - best_score_thread.inlier_number > 0) {
                        adapt = false;
                        supports.resize(std::min(num_tested_models, (int)supports.size()));
                        std::sort(supports.begin(), supports.end());
                        double delta = (double)supports[supports.size()/2] / points_size;
                        const double cdf_99 = delta * points_size + 2.32*sqrt(delta * points_size * (1 - delta));
                        delta = 0;
                        int num_lower = 0;
                        for (num_lower = 0; num_lower < (int)supports.size(); num_lower++) {
                            if (supports[num_lower] < cdf_99) delta += supports[num_lower];
                            else break;
                        }
                        delta /= (num_lower * points_size);
                        if (std::isnan(delta)) delta = params.getSPRTdelta();
                        min_non_random_inliers = (int)ceil(delta * points_size + 3.719*sqrt(delta * points_size * (1 - delta)));
                        model_verifier->updateSPRT(params.getTimeForModelEstimation(), 1, (double)mean_num_est_models / num_estimations, delta,
                                std::max((double)min_non_random_inliers/points_size, (double)best_score.inlier_number / points_size), best_score_all_threads);
                    }
                    if (!adapt && LO && num_hypothesis_tested < max_iters && !was_LO_run && !best_model_thread.empty())
                        runLO(iters);
                } // end of loop over iters
                if (! was_LO_run && !best_model_thread.empty() && LO)
                    runLO(-1 /*use full iterations of LO*/);

                best_model_thread.copyTo(best_models[thread_rng_id]);
                best_scores[thread_rng_id] = best_score_thread;
                num_tested_models_threads[thread_rng_id] = num_tested_models;
                tested_models_threads[thread_rng_id] = tested_models_thread;
                tested_samples_threads[thread_rng_id] = tested_samples_thread;
                best_samples_threads[thread_rng_id] = best_sample_thread;
            }}); // end parallel
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            // find best model from all threads' models
            best_score = best_scores[0];
            int best_thread_idx = 0;
            for (int i = 1; i < MAX_THREADS; i++) {
                if (best_scores[i].isBetter(best_score)) {
                    best_score = best_scores[i];
                    best_thread_idx = i;
                }
            }
            best_model = best_models[best_thread_idx];
            final_iters = num_hypothesis_tested;
            best_sample = best_samples_threads[best_thread_idx];
            for (int i = 0; i < MAX_THREADS; i++) {
                num_total_tested_models += num_tested_models_threads[i];
                if (models_for_random_test.size() < MAX_TEST_MODELS_NONRAND) {
                    for (int m = 0; m < tested_models_threads[i].size(); m++) {
                        models_for_random_test.emplace_back(tested_models_threads[i][m].clone());
                        samples_for_random_test.emplace_back(tested_samples_threads[i][m]);
                        if ((int)models_for_random_test.size() == MAX_TEST_MODELS_NONRAND)
                            break;
                    }
                }
            }
        }
        
        if (best_model.empty()) {
#if DEBUG
            std::cout << "BEST MODEL IS EMPTY!\n";
#endif
            ransac_output = ::vsac::Output(best_model, std::vector<bool>(), best_score.inlier_number, final_iters, true, std::vector<float>());
            return false;
        }
#if DEBUG
        std::cout << "IS LAST MODEL FROM LO " << last_model_from_LO << "\n";
#endif
       if (last_model_from_LO && params.isFundamental()) {
           Score new_score;
           Mat new_model;
           if (params.isParallel())
               _quality->getInliers(best_model, best_inliers_mask);
           if (_degeneracy.dynamicCast<FundamentalDegeneracy>()->verifyFundamental(best_model, best_score, best_inliers_mask, new_model, new_score)) {
#if DEBUG
                std::cout << "best model from LO is degenerate, score (" << best_score.score << ", " << best_score.inlier_number << "), new (" <<
                    new_score.score << "," << new_score.inlier_number <<"), sftb (" << best_score_model_not_from_LO.score << ", " << best_score_model_not_from_LO.inlier_number << ")" << "\n";
#endif
               if (new_score.isBetter(best_score) || new_score.isBetter(best_score_model_not_from_LO)) {
                   best_score = new_score;
                   new_model.copyTo(best_model);
               } else {
                   best_score = best_score_model_not_from_LO;
                   best_model_not_from_LO.copyTo(best_model);
               }
           }
       }

        // polish final model
#if DEBUG
        const auto temp_time = std::chrono::steady_clock::now();
#endif
       bool is_updated_by_polisher = false;
        if (params.getFinalPolisher() != ::vsac::PolishingMethod::NonePolisher) {
            Mat polished_model;
            Score polisher_score;
            if (params.isFundamental() && !params.isEnforceRank())
                _estimator->enforceRankConstraint(true); // now force rank constraint
            if (model_polisher->polishSoFarTheBestModel(best_model, best_score,
                    polished_model, polisher_score) && polisher_score.isBetter(best_score)) {
#if DEBUG
                polisher_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                std::cout << "POLISHER IS BETTER (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << polisher_score.score << ", " << polisher_score.inlier_number << ")\n";
#endif
                best_score = polisher_score;
                polished_model.copyTo(best_model);
                is_updated_by_polisher = true;
            }
#if DEBUG
            else {
                polisher_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                std::cout << "POLISHER IS WORSE (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << polisher_score.score << ", " << polisher_score.inlier_number << ")\n";
            }         
#endif
            // Only for F. if we dont force rank and model from LO (rank is not forced) and no polisher update, where rank is forced, then take a model from min sample
            if (params.isFundamental() && ! params.isEnforceRank() && ! is_updated_by_polisher && last_model_from_LO) {
#if DEBUG
                std::cout << "polisher score " << polisher_score.score << ", " << polisher_score.inlier_number << " is empty model " << polished_model.empty() << "\n";
#endif
                cv::Mat lo_model;
                Score lo_score;
                _local_optimization->setCurrentRANSACiter(-1);
                _local_optimization->refineModel(best_model, best_score, lo_model, lo_score);
                if (lo_score.isBetter(best_score) || lo_score.isBetter(polisher_score) || lo_score.isBetter(best_score_model_not_from_LO)) {
#if DEBUG
                    std::cout << "take best model from new lo (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << lo_score.score << ", " << lo_score.inlier_number << ")\n";
#endif
                    best_score = lo_score;
                    lo_model.copyTo(best_model);
                } else if (polisher_score.isBetter(best_score_model_not_from_LO)) {
#if DEBUG
                    std::cout << "take best model from polisher despite it worse (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << polisher_score.score << ", " << polisher_score.inlier_number << ")\n";
#endif
                    best_score = polisher_score;
                    polished_model.copyTo(best_model);
                } else {
#if DEBUG
                    std::cout << "take best model from min sample despite it worse (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << best_score_model_not_from_LO.score << ", " << best_score_model_not_from_LO.inlier_number << ")\n";
#endif
                    best_score = best_score_model_not_from_LO;
                    best_model_not_from_LO.copyTo(best_model);
                }
            }
        }

        // ================= here is ending ransac main implementation ===========================
        std::vector<bool> inliers_mask;
        std::vector<float> residuals;
        if (params.isMaskRequired()) {
            inliers_mask = std::vector<bool>(points_size);
            residuals = _quality->getErrorFnc()->getErrors(best_model);
            _quality->getInliers(residuals, inliers_mask, _quality->getThreshold());
        }

        bool is_random = false;
#if DEBUG
         const auto begin_time_test = std::chrono::steady_clock::now();
#endif
        if (params.isNonRandomnessTest()) {
            std::vector<int> temp_inliers(points_size);
            const int non_random_inls_best_model = getNumberOfNonRandomInliers(1, points, params.isFundamental(), best_model, true, best_sample,
                                                                                 temp_inliers, _quality->getInliers(best_model, temp_inliers));
             if (non_random_inls_best_model < 100) { // speed test, if more than 100 independent inliers then model is very likely to be non-random
                std::vector<float> inliers_list(models_for_random_test.size());

                std::vector<bool> temp_inls_mask(points_size);
                for (int m = 0; m < (int)models_for_random_test.size(); m++) {
                    const int inls = _quality->getInliers(models_for_random_test[m], temp_inliers);
                    _quality->getInliers(models_for_random_test[m], temp_inls_mask);
                    const int non_rand_inls = getNumberOfNonRandomInliers(1, points, params.isFundamental(), models_for_random_test[m],
                                                                          true, samples_for_random_test[m], temp_inliers, inls);
//                     std::cout << "number of non-rand inls " << m << ") " << non_rand_inls << " inls " << inls << " IoU " << Utils::intersectionOverUnion(inliers_mask, temp_inls_mask) << '\n';
//                    lambda_non_random += non_rand_inls;
                    inliers_list[m] = (float)non_rand_inls;
                }
                std::sort(inliers_list.begin(), inliers_list.end());
                double lambda_non_random_med = inliers_list.size() % 2 ? (inliers_list[inliers_list.size()/2] + inliers_list[inliers_list.size()/2+1])*0.5 : inliers_list[inliers_list.size()/2];
                // const int lambda_non_random_med = std::max(1., (double)Utils::findMedian(inliers_list));

                const double perc_95 = lambda_non_random_med + 1.644*sqrt(lambda_non_random_med * (1 - (double)lambda_non_random_med / (points_size - sample_size)));
                int lower_than_per = 0;
                double avg_lambda = 0;
                // std::cout <<"\n";
                for (const auto &inl : inliers_list) {
                    // std::cout << inl << " ";
                    if (inl < perc_95) {
                        avg_lambda += inl;
                        lower_than_per++;
                    }
                }
                // std::cout << "\n";
                avg_lambda = std::max(1., avg_lambda / lower_than_per);
//                lambda_non_random = std::max(1., lambda_non_random / (int)models_for_random_test.size());
//                const double cdf_lambda = Utils::getPoissonCDF(lambda_non_random, non_random_inls_best_model), cdf_N = pow(cdf_lambda, num_total_tested_models);
                const double cdf_lambda = Utils::getPoissonCDF(avg_lambda, non_random_inls_best_model), cdf_N = pow(cdf_lambda, num_total_tested_models);
                is_random = cdf_N < 0.9999;

                /////////////////// experiments /////////////////
                std::cout << final_iters << " & " << num_total_tested_models << " & " << non_random_inls_best_model << " & " << best_score.inlier_number << " & " <<
                    avg_lambda << " & " << lambda_non_random_med << " & " << perc_95 << " & " << (inliers_list.size() - lower_than_per) << " & " <<
                        // ut::str(cdf_lambda,4)
                        std::to_string(cdf_lambda).substr(0, std::to_string(cdf_lambda).find('.')+5)
                          << " & " << is_random << " & " ;
//                          << all_inliers_sample << " & " << ut::str(avg_num_inls_in_sample / inliers_list.size(),1) << " & ";
                /////////////////////////////////////////////////

//                 std::cout << "95% perc " << perc_95 << " higher than perc " << inliers_list.size()-lower_than_per << " new lambda " << avg_lambda << "\n";
//                 std::cout << "lambda median " << lambda_non_random_med << " total num of tested models " << num_total_tested_models << "\n";
//                 std::cout << "non random lambda " << lambda_non_random << " non random best model " << non_random_inls_best_model
//                           << " max inls ransac " << best_score.inlier_number << " cdf " << cdf_lambda << " tested models " << num_total_tested_models << " cdf^N " << cdf_N <<
//                           " final iters " << final_iters << " / " << params.getMaxIters() << " is random " << is_random << '\n';
             }
        }
//        std::cout << "time test " << std::chrono::duration_cast<std::chrono::microseconds>
//                (std::chrono::steady_clock::now() - begin_time_test).count() << "\n";
#if DEBUG
        std::cout << "iters " << final_iters << ", times: est " << est_time << " eval " << eval_time << " lo " << lo_time << 
            " degen time " << degensac_time << " pol time " << polisher_time << ", #LO " << num_lo_runs << " #SFTB " << num_so_far_the_best <<
            " #tested models " << num_total_tested_models << "ransac time " << std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::steady_clock::now() - begin_time).count() << '\n';
#endif
        ransac_output = ::vsac::Output(best_model, inliers_mask, best_score.inlier_number, final_iters, is_random, residuals);
        return true;
    }
};

int getNumberOfNonRandomInliers (double scale, const Mat &points, bool is_F, const Mat &model, bool has_sample,
        const std::vector<int> &sample, const std::vector<int> &inliers_, const int num_inliers_) {
    int num_inliers = num_inliers_;
    if (num_inliers == 0) return 0;
    std::vector<int> inliers= inliers_;
    const auto * const pts = (float *) points.data;
    const double ep_thr_sqr = 1e-6, line_thr = scale*0.01, /*5*/ neigh_thr_sqr = scale * 10; //scale * 9, 50
    int num_pts_bad_conditioning = 0, num_pts_near_ep = 0,
        num_pts_on_ep_lines = 0, num_pts_validatin_or_constr = 0, pt1 = 0;
    const int sample_size = is_F ? 7 : 4;
    double sign1 = 0, a1=0, b1=0, c1=0, a2=0, b2=0, c2=0, ep1_x, ep1_y, ep2_x, ep2_y;
    const auto * const m = (double *) model.data;
    Vec3d ep1;
    bool do_or_test = false, ep1_inf = false, ep2_inf = false;
    if (is_F) { // compute epipole and sign of the first point for orientation test
        ep1 = model.row(0).cross(model.row(2));
        auto * e = ep1.val;

        // e is zero vector, recompute e
        if (e[0] <= 1.9984e-15 && e[0] >= -1.9984e-15 &&
            e[1] <= 1.9984e-15 && e[1] >= -1.9984e-15 &&
            e[2] <= 1.9984e-15 && e[2] >= -1.9984e-15) {
            ep1 = model.row(1).cross(model.row(2));
        }

        cv::Vec3d ep2 = model.col(0).cross(model.col(2));
        e = ep2.val;

        // e is zero vector, recompute e
        if (e[0] <= 1.9984e-15 && e[0] >= -1.9984e-15 &&
            e[1] <= 1.9984e-15 && e[1] >= -1.9984e-15 &&
            e[2] <= 1.9984e-15 && e[2] >= -1.9984e-15) {
            ep2 = model.col(1).cross(model.col(2));
        }

        if (fabs(ep1[2]) < DBL_EPSILON) {
            ep1_inf = true; ep1_x = DBL_MAX; ep1_y = DBL_MAX;
        } else {
            ep1_x = ep1[0] / ep1[2];
            ep1_y = ep1[1] / ep1[2];
        }
        if (fabs(ep2[2]) < DBL_EPSILON) {
            ep2_inf = true; ep2_x = DBL_MAX; ep2_y = DBL_MAX;
        } else {
            ep2_x = ep2[0] / ep2[2];
            ep2_y = ep2[1] / ep2[2];
        }
    }
    const auto * const e1 = ep1.val; // of size 3x1

    // we move sample points to the end, so every inlier will be checked by sample point
    int num_sample_in_inliers;
    if (has_sample) {
        num_sample_in_inliers = 0;
        int temp_idx = num_inliers;
        for (int i = 0; i < temp_idx; i++) {
            const int inl = inliers[i];
            for (int s : sample) {
                if (inl == s) {
                    std::swap(inliers[i], inliers[--temp_idx]);
                    i--; // we need to check inlier that we just swapped
                    num_sample_in_inliers++;
                    break;
                }
            }
        }
        if (num_sample_in_inliers < sample_size)
            num_sample_in_inliers = sample_size; // for good model a few points does not matter, for bad could be decisive
    } else num_sample_in_inliers = sample_size;

    if (is_F) {
        int MIN_TEST = std::min(15, num_inliers);
        for (int i = 0; i < MIN_TEST; i++) {
            pt1 = 4*inliers[i];
            sign1 = (m[0]*pts[pt1+2]+m[3]*pts[pt1+3]+m[6])*(e1[1]-e1[2]*pts[pt1+1]);
            int validate = 0;
            for (int j = 0; j < MIN_TEST; j++) {
                if (i == j) continue;
                const int inl_idx = 4*inliers[j];
                if (sign1*(m[0]*pts[inl_idx+2]+m[3]*pts[inl_idx+3]+m[6])*(e1[1]-e1[2]*pts[inl_idx+1])<0) {
                    validate++;
                }
            }
            if (validate < MIN_TEST/2) {
                do_or_test = true;
                break;
            }
        }
    }

    // verification does not include sample points as they surely random
    const int max_verify = num_inliers - num_sample_in_inliers;
    if (max_verify <= 0)
        return 0;
    int num_non_random_inliers = max_verify;
    auto removeDependentPoints = [&] (bool do_orient_test, bool check_epipoles) {
        for (int i = 0; i < max_verify; i++) {
            // checks over inliers if they are dependent to other inliers
            const int inl_idx = 4*inliers[i];
            const double x1 = pts[inl_idx], y1 = pts[inl_idx+1], x2 = pts[inl_idx+2], y2 = pts[inl_idx+3];
            if (is_F) {
                // epipolar line on image 2 = l2
                a2 = m[0] * x1 + m[1] * y1 + m[2];
                b2 = m[3] * x1 + m[4] * y1 + m[5];
                c2 = m[6] * x1 + m[7] * y1 + m[8];
                // epipolar line on image 1 = l1
                a1 = m[0] * x2 + m[3] * y2 + m[6];
                b1 = m[1] * x2 + m[4] * y2 + m[7];
                c1 = m[2] * x2 + m[5] * y2 + m[8];
                if ((!ep1_inf && (x1-ep1_x)*(x1-ep1_x)+(y1-ep1_y)*(y1-ep1_y) < neigh_thr_sqr) ||
                    (!ep2_inf && (x2-ep2_x)*(x2-ep2_x)+(y2-ep2_y)*(y2-ep2_y) < neigh_thr_sqr)) {
                    num_non_random_inliers--;
                    num_pts_near_ep++;
                    continue; // is dependent, continue to the next point
                } else if (check_epipoles) {
                    if (a2 * a2 + b2 * b2 + c2 * c2 < ep_thr_sqr ||
                        a1 * a1 + b1 * b1 + c1 * c1 < ep_thr_sqr) {
                        num_non_random_inliers--;
                        num_pts_near_ep++;
                        continue; // is dependent, continue to the next point
                    }
                }
                else if (do_orient_test && pt1 != inl_idx && sign1*(m[0]*x2+m[3]*y2+m[6])*(e1[1]-e1[2]*y1)<0) {
                    num_non_random_inliers--;
                    num_pts_validatin_or_constr++;
                    continue;
                }
                const auto mag2 = 1 / sqrt(a2 * a2 + b2 * b2), mag1 = 1/sqrt(a1 * a1 + b1 * b1);
                a2 *= mag2; b2 *= mag2; c2 *= mag2;
                a1 *= mag1; b1 *= mag1; c1 *= mag1;
            }

            for (int j = i+1; j < num_inliers; j++) {// verify through all including sample points
                const int inl_idx_j = 4*inliers[j];
                const double X1 = pts[inl_idx_j], Y1 = pts[inl_idx_j+1], X2 = pts[inl_idx_j+2], Y2 = pts[inl_idx_j+3];
                const double dx1 = X1-x1, dy1 = Y1-y1, dx2 = X2-x2, dy2 = Y2-y2;
                if (dx1 * dx1 + dy1 * dy1 < neigh_thr_sqr || dx2 * dx2 + dy2 * dy2 < neigh_thr_sqr) {
                    num_non_random_inliers--;
                    num_pts_bad_conditioning++;
                    break; // is dependent stop verification
                } else if (is_F) {
                    if (fabs(a2 * X2 + b2 * Y2 + c2) < line_thr && //|| // xj'^T F   xi
                        fabs(a1 * X1 + b1 * Y1 + c1) < line_thr) { // xj^T  F^T xi'
                        num_non_random_inliers--;
                        num_pts_on_ep_lines++;
                        break; // is dependent stop verification
                    }
                }
            }
        }
    };
    removeDependentPoints(do_or_test, !ep1_inf && !ep2_inf);
    const bool is_pts_vald_constr_normal = (double)num_pts_validatin_or_constr / num_inliers < 0.6;
    const bool is_pts_near_ep_normal = (double)num_pts_near_ep / num_inliers < 0.6;
    if (!is_pts_near_ep_normal || !is_pts_vald_constr_normal) {
        num_non_random_inliers = max_verify;
        num_pts_bad_conditioning = 0; num_pts_near_ep = 0; num_pts_on_ep_lines = 0; num_pts_validatin_or_constr = 0;
        removeDependentPoints(is_pts_vald_constr_normal, is_pts_near_ep_normal);
    }
    return num_non_random_inliers;
}
}}

namespace vsac {
bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
               Output &output, cv::InputArray K1_, cv::InputArray K2_,
               cv::InputArray dist_coeff1, cv::InputArray dist_coeff2) {
    cv::Ptr<cv::vsac::NeighborhoodGraph> graph;
    cv::Ptr<cv::vsac::Error> error;
    cv::Ptr<cv::vsac::Estimator> estimator;
    cv::Ptr<cv::vsac::Degeneracy> degeneracy;
    cv::Ptr<cv::vsac::Quality> quality;
    cv::Ptr<cv::vsac::ModelVerifier> verifier;
    cv::Ptr<cv::vsac::Sampler> sampler;
    cv::Ptr<cv::vsac::RandomGenerator> lo_sampler;
    cv::Ptr<cv::vsac::Termination> termination;
    cv::Ptr<cv::vsac::LocalOptimization> lo;
    cv::Ptr<cv::vsac::FinalModelPolisher> polisher;
    cv::Ptr<cv::vsac::MinimalSolver> min_solver;
    cv::Ptr<cv::vsac::NonMinimalSolver> non_min_solver;
    cv::Ptr<cv::vsac::GammaValues> gamma_generator;

    int state = params.getRandomGeneratorState();
    cv::Mat points, K1, K2, calib_points, undist_points1, undist_points2, image_points;
    int points_size;
    double threshold = params.getThreshold(), max_thr = params.getMaximumThreshold();
    if (params.isPnP()) {
        if (! K1_.empty()) {
            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
            if (! dist_coeff1.empty()) {
                // undistortPoints also calibrate points using K
                if (points1.isContinuous())
                    undistortPoints(points1, undist_points1, K1_, dist_coeff1);
                else undistortPoints(points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
                points_size = cv::vsac::mergePoints(undist_points1, points2, points, true);
                cv::vsac::Utils::normalizeAndDecalibPointsPnP (K1, points, calib_points);
            } else {
                points_size = cv::vsac::mergePoints(points1, points2, points, true);
                cv::vsac::Utils::calibrateAndNormalizePointsPnP(K1, points, calib_points);
            }
        } else points_size = cv::vsac::mergePoints(points1, points2, points, true);
    } else {
        if (params.isEssential()) {
            CV_CheckEQ((int)(!K1_.empty() && !K2_.empty()), 1, "Intrinsic matrix must not be empty!");
            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
            K2 = K2_.getMat(); K2.convertTo(K2, CV_64F);
            if (! dist_coeff1.empty() || ! dist_coeff2.empty()) {
                // undistortPoints also calibrate points using K
                if (points1.isContinuous())
                    undistortPoints(points1, undist_points1, K1_, dist_coeff1);
                else undistortPoints(points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
                if (points2.isContinuous())
                    undistortPoints(points2, undist_points2, K2_, dist_coeff2);
                else undistortPoints(points2.getMat().clone(), undist_points2, K2_, dist_coeff2);
                points_size = cv::vsac::mergePoints(undist_points1, undist_points2, calib_points, false);
            } else {
                points_size = cv::vsac::mergePoints(points1, points2, points, false);
                cv::vsac::Utils::calibratePoints(K1, K2, points, calib_points);
            }
            threshold = cv::vsac::Utils::getCalibratedThreshold(threshold, K1, K2);
            max_thr = cv::vsac::Utils::getCalibratedThreshold(max_thr, K1, K2);
        } else {
            if (params.isFundamental()) {
                K1 = K1_.getMat(); K2 = K2_.getMat();
                if (! K1.empty() && ! K2.empty()) {
                    K1.convertTo(K1, CV_64F); K2.convertTo(K2, CV_64F);
                }
            }
            points_size = cv::vsac::mergePoints(points1, points2, points, false);
        }
    }

    // Since error function output squared error distance, so make
    // threshold squared as well
    threshold *= threshold;

    //todo: add inner_thre
    if (params.getSampler() == SamplingMethod::SAMPLING_NAPSAC || params.getLO() == LocalOptimMethod::LOCAL_OPTIM_GC) {
        if (params.getNeighborsSearch() == NeighborSearchMethod::NEIGH_GRID) {
            graph = cv::vsac::GridNeighborhoodGraph::create(points, points_size,
                    params.getCellSize(), params.getCellSize(), params.getCellSize(), params.getCellSize(), 10);
        } else if (params.getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_KNN) {
            graph = cv::vsac::FlannNeighborhoodGraph::create(points, points_size,params.getKNN(), false, 5, 1);
        } else if (params.getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_RADIUS) {
            assert(false && "check if it RadiusSearchNeighborhoodGraph works properly!");
            graph = cv::vsac::RadiusSearchNeighborhoodGraph::create(points, points_size,
                                                                    params.getGraphRadius(), 5, 1);
        } else CV_Error(cv::Error::StsNotImplemented, "Graph type is not implemented!");
    }

    std::vector<cv::Ptr<cv::vsac::NeighborhoodGraph>> layers;
    if (params.getSampler() == SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
        CV_CheckEQ((int)params.isPnP(), 0, "ProgressiveNAPSAC for PnP is not implemented!");
        const auto &cell_number_per_layer = params.getGridCellNumber();
        layers.reserve(cell_number_per_layer.size());
        const auto * const pts = (float *) points.data;
        float img1_width = 0, img1_height = 0, img2_width = 0, img2_height = 0;
        for (int i = 0; i < 4 * points_size; i += 4) {
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
            layers.emplace_back(cv::vsac::GridNeighborhoodGraph::create(points, points_size,
        (int)(img1_width / (float)cell_number), (int)(img1_height / (float)cell_number),
        (int)(img2_width / (float)cell_number), (int)(img2_height / (float)cell_number), 10));
        }
    }

    // update points by calibrated for Essential matrix after graph is calculated
    if (params.isEssential()) {
        points.copyTo(image_points);
        points = calib_points;
        // if maximum calibrated threshold significanlty differs threshold then set upper bound
        if (max_thr > 10*threshold)
            max_thr = sqrt(10*threshold); // max thr will be squared after
    }
    if (max_thr < threshold)
        max_thr = threshold;

    std::vector<std::vector<int>> close_pts_mask;
    initialize (state, points_size, threshold, max_thr, params, points, calib_points, image_points, layers, close_pts_mask, K1, K2, graph, min_solver,
                non_min_solver, gamma_generator, error, estimator, degeneracy, quality, verifier, lo, termination, sampler, lo_sampler, false);

    cv::Ptr<cv::vsac::RandomGenerator> polisher_sampler;
    cv::Ptr<cv::vsac::CovarianceSolver> cov_polisher;
    switch (params.getFinalPolisher()) {
        case PolishingMethod::CovPolisher:
            if (params.isFundamental()) {
                cov_polisher = cv::vsac::CovarianceEpipolarSolver::create(points, true);
            } else if (params.isHomography()) {
                cov_polisher = cv::vsac::CovarianceHomographySolver::create(points);
            } else if (params.isEssential()) {
                cov_polisher = cv::vsac::CovarianceEpipolarSolver::create(calib_points, false);
                if (! params.isEnforceRank())
                    cov_polisher->setEnforceRankConstraint(false);
            } else if (params.getEstimator() == EstimationMethod::Affine) {
                cov_polisher = cv::vsac::CovarianceAffineSolver::create(points);
            } else if (params.isPnP()) {
                polisher = cv::vsac::LeastSquaresPolishing::create(estimator, quality, params.getFinalLSQIterations()); // use lsq polisher here
                break;
            } else assert(false && "covariance polisher not implemented\n");
            polisher = cv::vsac::CovariancePolisher::create(degeneracy, quality, cov_polisher, params.getFinalLSQIterations());
            break;
        case PolishingMethod::MAGSAC:
            if (gamma_generator == nullptr)
                gamma_generator = cv::vsac::GammaValues::create(params.getDegreesOfFreedom());
            polisher = cv::vsac::WeightedPolisher::create(degeneracy, quality, non_min_solver, quality->getErrorFnc(), gamma_generator, 10, params.getDegreesOfFreedom(), params.getUpperIncompleteOfSigmaQuantile(),
                                                          params.getC(), 2*max_thr);
            break;
        case PolishingMethod ::LSQPolisher:
            polisher = cv::vsac::LeastSquaresPolishing::create(estimator, quality, params.getFinalLSQIterations());
            break;
        case PolishingMethod ::IterativePolish:
            polisher_sampler = cv::vsac::UniformRandomGenerator::create(state++, points_size, params.getLOSampleSize());
            polisher = cv::vsac::IterativePolisher::create(quality, degeneracy, estimator, polisher_sampler, 20);
            break;
        default : break;
    }

    const bool is_parallel = params.isParallel();
    cv::vsac::VSAC ransac (points, params, points_size, estimator, quality, sampler,
                           termination, verifier, degeneracy, lo, polisher, is_parallel, state);
    if (is_parallel)
        ransac.setDataForParallel (threshold, max_thr, K1, K2, calib_points, image_points, graph, layers, close_pts_mask);
    return ransac.run(output);
}
}
