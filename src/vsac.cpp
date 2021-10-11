#include "precomp.hpp"
#include <atomic>

namespace cv { namespace vsac {
bool VSAC::run(::vsac::Output &vsac_output) {
    if (points_sample_size < params.getSampleSize())
        return false;
#ifdef DEBUG
    const auto begin_time = std::chrono::steady_clock::now();
#endif
    const bool LO = params.getLO() != ::vsac::LocalOptimMethod::LOCAL_OPTIM_NULL, IS_QUASI_SAMPLING = params.isQuasiSampling(),
        IS_FUNDAMENTAL = params.isFundamental(), IS_NON_RAND_TEST = params.isNonRandomnessTest();
    const int MAX_MODELS_ADAPT = 21, MAX_ITERS_ADAPT = MAX_MODELS_ADAPT/*assume at least 1 model from 1 sample*/,
        sample_size = params.getSampleSize(), MAX_ITERS_BEFORE_LO = params.getMaxItersBeforeLO();
    const double IOU_SIMILARITY_THR = 0.80;
    std::vector<int> non_degen_sample, best_sample;

    double lambda_non_random_all_inliers = -1;
    int final_iters, num_total_tested_models = 0;

#ifdef DEBUG
    double degensac_time = 0, est_time = 0, eval_time = 0, polisher_time = 0, lo_time = 0, num_degenerate_cases = 0, num_degensac_runs = 0, sample_time = 0, sftb_time = 0;
#endif

    // non-random
    const int MAX_TEST_MODELS_NONRAND = IS_NON_RAND_TEST ? MAX_MODELS_ADAPT : 0;
    std::vector<Mat> models_for_random_test; models_for_random_test.reserve(MAX_TEST_MODELS_NONRAND);
    std::vector<std::vector<int>> samples_for_random_test; samples_for_random_test.reserve(MAX_TEST_MODELS_NONRAND);

    bool last_model_from_LO = false;
    Mat best_model, best_model_not_from_LO, K1_approx, K2_approx;
    Score best_score, best_score_model_not_from_LO;
    std::vector<bool> best_inliers_mask(points_val_size);
    if (! parallel) {
        // adaptive sprt test
        double IoU = 0, mean_time_estimation = 0, mean_num_est_models = 0, mean_time_evaluation = 0;
        bool adapt = IS_NON_RAND_TEST || params.getVerifier() == ::vsac::VerificationMethod ::ASPRT, was_LO_run = false;
        int min_non_random_inliers = 0, iters = 0, max_iters = params.getMaxIters(), num_estimations = 0;
        Mat non_degenerate_model, lo_model;
        Score current_score, lo_score, non_degenerate_model_score;
        std::vector<bool> model_inliers_mask (points_val_size);
        std::vector<Mat> models(_estimator->getMaxNumSolutions());
        std::vector<int> sample(_estimator->getMinimalSampleSize()), supports;
        supports.reserve(3*MAX_MODELS_ADAPT); // store model supports during adaption
        auto update_best = [&] (const Mat &new_model, const Score &new_score, bool from_lo=false) {
#ifdef DEBUG
            const auto sftb_begin_time = std::chrono::steady_clock::now();
#endif
            _quality->getInliers(new_model, model_inliers_mask);
            if (!adapt && IS_QUASI_SAMPLING && new_score.inlier_number > 100)// update quasi sampler
                _sampler->updateSampler(model_inliers_mask);
            // IoU is used for LO and adaption
            IoU = Utils::intersectionOverUnion(best_inliers_mask, model_inliers_mask);
#ifdef DEBUG
             std::cout << "UPDATE BEST, iters " << iters << " (" << best_score.score << "," << best_score.inlier_number << ") -> (" << new_score.score << ", " << new_score.inlier_number << ") IoU " << IoU << " from LO " << from_lo << '\n';
#endif
            if (!best_model.empty() && models_for_random_test.size() < MAX_TEST_MODELS_NONRAND && IoU < IOU_SIMILARITY_THR) { // use IoU to not save similar models
                // save old best model for non-randomness test if necessary
                models_for_random_test.emplace_back(best_model.clone());
                samples_for_random_test.emplace_back(best_sample);
            }
            if (!adapt) { // update quality and verifier to save evaluation time of a model
                _quality->setBestScore(new_score.score);
                _model_verifier->update(new_score.inlier_number);
            }
            // update score, model, inliers and max iterations
            best_inliers_mask = model_inliers_mask;
            best_score = new_score;
            new_model.copyTo(best_model);
            best_sample = sample;
            max_iters = _termination->update(best_model, quality_sample->getScore(best_model).inlier_number);
            if (IS_FUNDAMENTAL) { // avoid degeneracy after LO run
                last_model_from_LO = from_lo;
                if (!last_model_from_LO) {
                    // save last model not from LO
                    best_model.copyTo(best_model_not_from_LO);
                    best_score_model_not_from_LO = best_score;
                }
            }
#ifdef DEBUG
            sftb_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sftb_begin_time).count();
#endif
        };
        auto runLO = [&] (int current_ransac_iters) {
#ifdef DEBUG
            const auto temp_time = std::chrono::steady_clock::now();
#endif
            was_LO_run = true;
            _local_optimization->setCurrentRANSACiter(current_ransac_iters);
            if (_local_optimization->refineModel
                    (best_model, best_score, lo_model, lo_score) && lo_score.isBetter(best_score)){
#ifdef DEBUG
                lo_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                std::cout << "START LO at " << iters << " BEST (" << best_score.score << "," << best_score.inlier_number << "), LO is BETTER (" << lo_score.score << ", " << lo_score.inlier_number << ")\n";
#endif
                update_best(lo_model, lo_score, true);
            }
#ifdef DEBUG
            else {
                lo_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                 std::cout << "START LO at " << iters << " BEST (" << best_score.score << "," << best_score.inlier_number << "), LO is WORSE (" << lo_score.score << ", " << lo_score.inlier_number << ")\n";
            }
#endif
        };
        for (; iters < max_iters; iters++) {
#ifdef DEBUG
            const auto sample_begin_time = std::chrono::steady_clock::now();
#endif
            _sampler->generateSample(sample);
#ifdef DEBUG
            sample_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sample_begin_time).count();
#endif
            int number_of_models;
            if (adapt) {
                const auto time_estimation = std::chrono::steady_clock::now();
                number_of_models = _estimator->estimateModels(sample, models);
#ifdef DEBUG
                est_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_estimation).count();
#endif
                if (iters != 0)
                    mean_time_estimation += std::chrono::duration_cast<std::chrono::microseconds>
                        (std::chrono::steady_clock::now() - time_estimation).count();
                mean_num_est_models += number_of_models;
                num_estimations++;
            } else {
#ifdef DEBUG
                const auto temp_time = std::chrono::steady_clock::now();
#endif
                number_of_models = _estimator->estimateModels(sample, models);
#ifdef DEBUG
                est_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
#endif
            }
            for (int i = 0; i < number_of_models; i++) {
                num_total_tested_models++;
                if (adapt) {
                    const auto time_evaluation = std::chrono::steady_clock::now();
                    current_score = _quality->getScore(models[i]);
                    if (iters != 0)
                        mean_time_evaluation += std::chrono::duration_cast<std::chrono::microseconds>
                            (std::chrono::steady_clock::now() - time_evaluation).count();
#ifdef DEBUG
                    eval_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_evaluation).count();
#endif
                    supports.emplace_back(current_score.inlier_number);
                } else {
#ifdef DEBUG
                    const auto temp_time = std::chrono::steady_clock::now();
#endif
                    if (_model_verifier->isModelGood(models[i])) {
                        if (!_model_verifier->getScore(current_score))
                            current_score = _quality->getScore(models[i]);
                    } else {
#ifdef DEBUG
                        eval_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
                        const auto sc = _quality->getScore(models[i]);
                        if (sc.isBetter(best_score))
                            std::cout << "SPRT REJECTED BETTER MODEL (" << sc.score << ", " << sc.inlier_number << ")\n";
#endif
                        continue;
                    }
#ifdef DEBUG
                    eval_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
#endif
                }
#ifdef DEBUG
                std::cout << "iter " << iters << " score (" << current_score.score << ", " << current_score.inlier_number << ")\n";
#endif
                if (current_score.isBetter(best_score)) {
#ifdef DEBUG
                    const auto temp_time = std::chrono::steady_clock::now();
                    const bool is_degen = _degeneracy->recoverIfDegenerate(sample, models[i], current_score, non_degenerate_model, non_degenerate_model_score);
                    degensac_time += std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now() - temp_time).count();
                    num_degensac_runs++;
                    if (is_degen) {
                        num_degenerate_cases++;
#else
                    if (_degeneracy->recoverIfDegenerate(sample, models[i], current_score,
                               non_degenerate_model, non_degenerate_model_score)) {
#endif
#ifdef DEBUG
                        std::cout << "IS DEGENERATE, new score (" << non_degenerate_model_score.score << ", " << non_degenerate_model_score.inlier_number << ")\n";
#endif
                        // check if best non degenerate model is better than so far the best model
                        if (non_degenerate_model_score.isBetter(best_score))
                            update_best(non_degenerate_model, non_degenerate_model_score);
                        else continue;
                    } else update_best(models[i], current_score);
                    if (!adapt && iters < max_iters && LO && best_score.inlier_number > min_non_random_inliers && IoU < IOU_SIMILARITY_THR && iters > MAX_ITERS_BEFORE_LO)
                        runLO(iters); // update model by Local optimization
                } // end of if so far the best score
                else if (models_for_random_test.size() < MAX_TEST_MODELS_NONRAND) {
                    models_for_random_test.emplace_back(models[i].clone());
                    samples_for_random_test.emplace_back(sample);
                }
                if (iters > max_iters)
                    break; // break loop over models
            } // end loop of number of models
            if (adapt && iters >= MAX_ITERS_ADAPT && num_total_tested_models >= MAX_MODELS_ADAPT) {
                adapt = false;
                lambda_non_random_all_inliers = getLambda(supports, 2.32, points_val_size, sample_size, false, min_non_random_inliers);
#ifdef DEBUG
                std::cout << "ADAPT ENDED, iters " << iters << " models " << num_total_tested_models << " min non-rand inls " << min_non_random_inliers << ", delta " << delta << " mu (lambda) " << delta * points_size << "\n";
#endif
                _model_verifier->updateSPRT(mean_time_estimation/(num_estimations-1), mean_time_evaluation/((num_total_tested_models-1) * points_val_size),
                mean_num_est_models/num_estimations, lambda_non_random_all_inliers/points_val_size,(double)std::max(min_non_random_inliers, best_score.inlier_number)/points_val_size, best_score);
            }
            if (!adapt && LO && iters < max_iters && !best_model.empty() && !was_LO_run &&
                    best_score.inlier_number > min_non_random_inliers && iters > MAX_ITERS_BEFORE_LO)
                runLO(iters);
        } // end main while loop
        final_iters = iters;
        if (! was_LO_run && !best_model.empty() && LO)
            runLO(-1 /*use full iterations of LO*/);
    } else { // parallel VSAC
        const int MAX_THREADS = getNumThreads(), growth_max_samples = 200000;
        const bool is_prosac = params.getSampler() == ::vsac::SamplingMethod::SAMPLING_PROSAC;
        std::atomic_bool success(false);
        std::atomic_int num_hypothesis_tested(0), thread_cnt(0), max_number_inliers(0), subset_size, termination_length;
        std::atomic<double> best_score_all(std::numeric_limits<double>::max());
        std::vector<Score> best_scores(MAX_THREADS), best_scores_not_LO;
        std::vector<Mat> best_models(MAX_THREADS), best_models_not_LO, K1_apx, K2_apx;
        std::vector<int> num_tested_models_threads(MAX_THREADS), growth_function, non_random_inliers;
        std::vector<std::vector<Mat>> tested_models_threads(MAX_THREADS);
        std::vector<std::vector<std::vector<int>>> tested_samples_threads(MAX_THREADS);
        std::vector<std::vector<int>> best_samples_threads(MAX_THREADS);
        std::vector<bool> last_model_from_LO_vec;
        std::vector<double> lambda_non_random_all_inliers_vec(MAX_THREADS);
        if (IS_FUNDAMENTAL) {
            last_model_from_LO_vec = std::vector<bool>(MAX_THREADS);
            best_models_not_LO = std::vector<Mat>(MAX_THREADS);
            best_scores_not_LO = std::vector<Score>(MAX_THREADS);
            K1_apx = std::vector<Mat>(MAX_THREADS);
            K2_apx = std::vector<Mat>(MAX_THREADS);
        }
        if (is_prosac) {
            growth_function = _sampler.dynamicCast<ProsacSampler>()->getGrowthFunction();
            subset_size = 2*sample_size; // n,  size of the current sampling pool
            termination_length = points_sample_size;
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////////////
        parallel_for_(Range(0, MAX_THREADS), [&](const Range & /*range*/) {
        if (!success) { // cover all if not success to avoid thread creating new variables
            const int thread_rng_id = thread_cnt++;
            bool adapt = params.getVerifier() == ::vsac::VerificationMethod ::ASPRT || IS_NON_RAND_TEST;
            int thread_state = _state + thread_rng_id, min_non_random_inliers = 0, num_tested_models = 0,
                num_estimations = 0, mean_num_est_models = 0, iters, max_iters = params.getMaxIters();
            double IoU = 0, lambda_non_random_all_inliers_thread = -1;
            std::vector<Mat> tested_models_thread; tested_models_thread.reserve(MAX_TEST_MODELS_NONRAND);
            std::vector<std::vector<int>> tested_samples_thread; tested_samples_thread.reserve(MAX_TEST_MODELS_NONRAND);
            Ptr<UniformRandomGenerator> random_gen;
            if (is_prosac) random_gen = UniformRandomGenerator::create(thread_state);
            Ptr<Error> error_val, error_sample;
            Ptr<Estimator> estimator;
            Ptr<Degeneracy> degeneracy;
            Ptr<Quality> quality, quality_sample_thread;
            Ptr<ModelVerifier> model_verifier;
            Ptr<Sampler> sampler;
            Ptr<RandomGenerator> lo_sampler;
            Ptr<Termination> termination, lo_termination;
            Ptr<LocalOptimization> local_optimization;
            Ptr<MinimalSolver> min_solver;
            Ptr<NonMinimalSolver> non_min_solver;
            Ptr<GammaValues> gamma_generator;
            initialize (thread_state, min_solver, non_min_solver, gamma_generator, error_sample, error_val, estimator, degeneracy, quality,
                    model_verifier, local_optimization, termination, sampler, lo_sampler, lo_termination, true);
            quality_sample_thread = RansacQuality::create(points_sample_size, threshold, error_sample);
            bool is_last_from_LO_thread = false;
            Mat best_model_thread, non_degenerate_model, lo_model, best_not_LO_thread;
            Score best_score_thread, current_score, non_denegenerate_model_score, lo_score,best_score_all_threads, best_not_LO_score_thread;
            std::vector<int> sample(estimator->getMinimalSampleSize()), best_sample_thread, supports;
            supports.reserve(3*MAX_MODELS_ADAPT); // store model supports
            std::vector<bool> best_inliers_mask(points_val_size, false), model_inliers_mask(points_val_size, false);
            std::vector<Mat> models(estimator->getMaxNumSolutions());
            auto update_best = [&] (const Score &new_score, const Mat &new_model, bool from_LO=false) {
                // update best score of all threads
                if (max_number_inliers < new_score.inlier_number) max_number_inliers = new_score.inlier_number;
                if (best_score_all > new_score.score) best_score_all = new_score.score;
                best_score_all_threads = Score(max_number_inliers, best_score_all);
                //
                quality->getInliers(new_model, model_inliers_mask);
                if (!adapt && IS_QUASI_SAMPLING && new_score.inlier_number > 100)
                    sampler->updateSampler(model_inliers_mask);
                IoU = Utils::intersectionOverUnion(best_inliers_mask, model_inliers_mask);
                if (!best_model_thread.empty() && tested_models_thread.size() < MAX_TEST_MODELS_NONRAND && IoU < IOU_SIMILARITY_THR) {
                    tested_models_thread.emplace_back(best_model_thread.clone());
                    tested_samples_thread.emplace_back(best_sample_thread);
                }
                if (!adapt) { // update quality and verifier
                    quality->setBestScore(best_score_all);
                    model_verifier->update(max_number_inliers);
                }
                // copy new score to best score
                best_score_thread = new_score;
                best_sample_thread = sample;
                best_inliers_mask = model_inliers_mask;
                // remember best model
                new_model.copyTo(best_model_thread);

                // update upper bound of iterations
                if (is_prosac) {
                    int new_termination_length;
                    max_iters = termination.dynamicCast<ProsacTerminationCriteria>()->
                            updateTerminationLength(best_model_thread, quality_sample_thread->getScore(best_model_thread).inlier_number, new_termination_length);
                    // update termination length
                    if (new_termination_length < termination_length)
                        termination_length = new_termination_length;
                } else max_iters = termination->update(best_model_thread, max_number_inliers);
                if (IS_FUNDAMENTAL) {
                    is_last_from_LO_thread = from_LO;
                    if (!from_LO) {
                        best_model_thread.copyTo(best_not_LO_thread);
                        best_not_LO_score_thread = best_score_thread;
                    }
                }
            };
            bool was_LO_run = false;
            auto runLO = [&] (int current_ransac_iters) {
                was_LO_run = true;
                local_optimization->setCurrentRANSACiter(current_ransac_iters);
                if (local_optimization->refineModel(best_model_thread, best_score_thread, lo_model, 
                        lo_score) && lo_score.isBetter(best_score_thread))
                    update_best(lo_score, lo_model, true);
            };
            for (iters = 0; iters < max_iters && !success; iters++) {
                success = num_hypothesis_tested++ > max_iters;
                if (iters % 10 && !adapt) {
                    // Synchronize threads. just to speed verification of model.
                    quality->setBestScore(std::min(best_score_thread.score, (double)best_score_all));
                    model_verifier->update(std::max(best_score.inlier_number, (int)max_number_inliers));
                }

                if (is_prosac) {
                    if (num_hypothesis_tested > growth_max_samples) {
                        // if PROSAC has not converged to solution then do uniform sampling.
                        random_gen->generateUniqueRandomSet(sample, sample_size, points_sample_size);
                    } else {
                        if (num_hypothesis_tested >= growth_function[subset_size-1] && subset_size < termination_length-MAX_THREADS) {
                            subset_size++;
                            if (subset_size >= points_sample_size) subset_size = points_sample_size-1;
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
                    num_estimations++; mean_num_est_models += number_of_models;
                }
                for (int i = 0; i < number_of_models; i++) {
                    num_tested_models++;
                    if (adapt) {
                        current_score = quality->getScore(models[i]);
                        supports.emplace_back(current_score.inlier_number);
                    } else if (model_verifier->isModelGood(models[i])) {
                        if (!model_verifier->getScore(current_score))
                            current_score = quality->getScore(models[i]);
                    } else continue;

                    if (current_score.isBetter(best_score_all_threads)) {
                        if (degeneracy->recoverIfDegenerate(sample, models[i], current_score,
                                non_degenerate_model, non_denegenerate_model_score)) {
                            // check if best non degenerate model is better than so far the best model
                            if (non_denegenerate_model_score.isBetter(best_score_thread))
                                update_best(non_denegenerate_model_score, non_degenerate_model);
                            else continue;
                        } else update_best(current_score, models[i]);
                        if (!adapt && LO && num_hypothesis_tested < max_iters && IoU < IOU_SIMILARITY_THR &&
                                best_score_thread.inlier_number > min_non_random_inliers && num_hypothesis_tested > MAX_ITERS_BEFORE_LO)
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
                if (adapt && iters >= MAX_ITERS_ADAPT && num_tested_models >= MAX_MODELS_ADAPT) {
                    adapt = false;
                    lambda_non_random_all_inliers_thread = getLambda(supports, 2.32, points_val_size, sample_size, false, min_non_random_inliers);
                    model_verifier->updateSPRT(params.getTimeForModelEstimation(), 1, (double)mean_num_est_models/num_estimations, lambda_non_random_all_inliers_thread/points_val_size,
                         (double)std::max(min_non_random_inliers, best_score.inlier_number)/points_val_size, best_score_all_threads);
                }
                if (!adapt && LO && num_hypothesis_tested < max_iters && !was_LO_run && !best_model_thread.empty() &&
                        best_score_thread.inlier_number > min_non_random_inliers && num_hypothesis_tested > MAX_ITERS_BEFORE_LO)
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
            if (IS_FUNDAMENTAL) {
                degeneracy.dynamicCast<FundamentalDegeneracy>()->getApproximatedIntrinsics(K1_apx[thread_rng_id], K2_apx[thread_rng_id]);
                best_scores_not_LO[thread_rng_id] = best_not_LO_score_thread;
                best_not_LO_thread.copyTo(best_models_not_LO[thread_rng_id]);
                last_model_from_LO_vec[thread_rng_id] = is_last_from_LO_thread;
            }
            lambda_non_random_all_inliers_vec[thread_rng_id] = lambda_non_random_all_inliers_thread;
        }}); // end parallel
        ///////////////////////////////////////////////////////////////////////////////////////////////////////
        // find best model from all threads' models
        best_score = best_scores[0];
        int best_thread_idx = 0;
        for (int i = 1; i < MAX_THREADS; i++)
            if (best_scores[i].isBetter(best_score)) {
                best_score = best_scores[i];
                best_thread_idx = i;
            }
        best_model = best_models[best_thread_idx];
        if (IS_FUNDAMENTAL) {
            last_model_from_LO = last_model_from_LO_vec[best_thread_idx];
            K1_approx = K1_apx[best_thread_idx];
            K2_approx = K2_apx[best_thread_idx];
        }
        final_iters = num_hypothesis_tested;
        best_sample = best_samples_threads[best_thread_idx];
        int num_lambdas = 0;
        double avg_lambda = 0;
        for (int i = 0; i < MAX_THREADS; i++) {
            if (IS_FUNDAMENTAL && best_scores_not_LO[i].isBetter(best_score_model_not_from_LO)) {
                best_score_model_not_from_LO = best_scores_not_LO[i];
                best_models_not_LO[i].copyTo(best_model_not_from_LO);
            }
            if (IS_NON_RAND_TEST && lambda_non_random_all_inliers_vec[i] > 0) {
                num_lambdas ++;
                avg_lambda += lambda_non_random_all_inliers_vec[i];
            }
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
        if (IS_NON_RAND_TEST && num_lambdas > 0 && avg_lambda > 0)
            lambda_non_random_all_inliers = avg_lambda / num_lambdas;
    }

    if (best_model.empty()) {
#ifdef DEBUG
        std::cout << "BEST MODEL IS EMPTY!\n";
#endif
                vsac_output = ::vsac::Output(best_model, std::vector<bool>(), best_score.inlier_number, final_iters, ::vsac::MODEL_CONFIDENCE::RANDOM, std::vector<float>());
        return false;
    }
#ifdef DEBUG
//    std::cout << "IS LAST MODEL FROM LO " << last_model_from_LO << "\n";
    const auto begin_last_verify = std::chrono::steady_clock::now();
#endif
   if (last_model_from_LO && IS_FUNDAMENTAL) {
       Score new_score; Mat new_model;
       if (parallel)
           _quality->getInliers(best_model, best_inliers_mask);
       // run additional degeneracy check for F:
       if (_degeneracy.dynamicCast<FundamentalDegeneracy>()->verifyFundamental(best_model, best_score, best_inliers_mask, new_model, new_score)) {
#ifdef DEBUG
            std::cout << "best model from LO is degenerate, score (" << best_score.score << ", " << best_score.inlier_number << "), new (" <<
                new_score.score << "," << new_score.inlier_number <<"), sftb (" << best_score_model_not_from_LO.score << ", " << best_score_model_not_from_LO.inlier_number << ")" << "\n";
#endif
            // so-far-the-best F is degenerate
            // Update best F using non-degenerate F or the one which is not from LO
           if (new_score.isBetter(best_score_model_not_from_LO)) {
               best_score = new_score;
               new_model.copyTo(best_model);
           } else {
               best_score = best_score_model_not_from_LO;
               best_model_not_from_LO.copyTo(best_model);
           }
       } else { // so-far-the-best F is not degenerate
           if (new_score.isBetter(best_score)) {
                // if new model is better then update
               best_score = new_score;
               new_model.copyTo(best_model);
           }
       }
   }

#ifdef DEBUG
    degensac_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_last_verify).count();
    const auto temp_time = std::chrono::steady_clock::now();
#endif
    if (params.getFinalPolisher() != ::vsac::PolishingMethod::NonePolisher) {
        Mat polished_model;
        Score polisher_score;
        if (polisher->polishSoFarTheBestModel(best_model, best_score, // polish final model
              polished_model, polisher_score) && polisher_score.isBetter(best_score)) {
#ifdef DEBUG
            polisher_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
//            std::cout << "POLISHER IS BETTER (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << polisher_score.score << ", " << polisher_score.inlier_number << ")\n";
#endif
            best_score = polisher_score;
            polished_model.copyTo(best_model);
        }
#ifdef DEBUG
        else {
            polisher_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - temp_time).count();
//            std::cout << "POLISHER IS WORSE (" << best_score.score << ", " << best_score.inlier_number << ") -> (" << polisher_score.score << ", " << polisher_score.inlier_number << ")\n";
        }
#endif
    }

    ///////////////// get inliers of the best model and points' residuals ///////////////
    std::vector<bool> inliers_mask; std::vector<float> residuals;
    if (params.isMaskRequired()) {
        inliers_mask = std::vector<bool>(points_val_size);
        residuals = _error_val->getErrors(best_model);
        _quality->getInliers(residuals, inliers_mask, threshold);
    }

    ::vsac::MODEL_CONFIDENCE model_conf = ::vsac::MODEL_CONFIDENCE::UNKNOWN;
#ifdef DEBUG
     const auto begin_time_test = std::chrono::steady_clock::now();
#endif
    if (IS_NON_RAND_TEST) {
        std::vector<int> temp_inliers(points_val_size);
        const int non_random_inls_best_model = getIndependentInliers(best_model, best_sample, temp_inliers,
                     _quality->getInliers(best_model, temp_inliers));
        // quick test on lambda from all inliers (= upper bound of independent inliers)
        // if model with independent inliers is not random for Poisson with all inliers then it is not random using independent inliers too
        if (pow(Utils::getPoissonCDF(lambda_non_random_all_inliers, non_random_inls_best_model), num_total_tested_models) < 0.9999) {
            std::vector<int> inliers_list(models_for_random_test.size());
            for (int m = 0; m < (int)models_for_random_test.size(); m++)
                inliers_list[m] = getIndependentInliers(models_for_random_test[m], samples_for_random_test[m],
                    temp_inliers, _quality->getInliers(models_for_random_test[m], temp_inliers));
            int min_non_rand_inliers;
            const double lambda = getLambda(inliers_list, 1.644, points_val_size, sample_size, true, min_non_rand_inliers);
            const double cdf_lambda = Utils::getPoissonCDF(lambda, non_random_inls_best_model), cdf_N = pow(cdf_lambda, num_total_tested_models);
            model_conf = cdf_N < 0.9999 ? ::vsac::MODEL_CONFIDENCE ::RANDOM : ::vsac::MODEL_CONFIDENCE ::NON_RANDOM;
#ifdef DEBUG
            std::cout << "iters " << final_iters << " tested models " << num_total_tested_models << " ind best " << non_random_inls_best_model << " #inls " << best_score.inlier_number
                      << " l " << lambda << " is random " << is_random << " & " ;
#endif
        } else model_conf = ::vsac::MODEL_CONFIDENCE ::NON_RANDOM;
    }
#ifdef DEBUG
    const auto nonrand_test = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_time_test).count();
    const auto ransac_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_time).count();
    double time_main_components = est_time + eval_time + lo_time + degensac_time + polisher_time + nonrand_test + sample_time + sftb_time;
    std::cout << "iter " << final_iters << ", time: est " << est_time << " eval " << eval_time << " lo " << lo_time <<
        " deg " << degensac_time << " pol " << polisher_time << " sample " << sample_time << " sftb " << sftb_time <<
        " #tested m " << num_total_tested_models << " non-rand " << nonrand_test << " ransac " << ransac_time << " mains " << time_main_components << '\n';
#endif
    vsac_output = ::vsac::Output(best_model, inliers_mask, best_score.inlier_number, final_iters, model_conf, residuals);
    if (IS_FUNDAMENTAL) {
        if (parallel) {
            vsac_output.K1 = K1_approx.clone(); vsac_output.K2 = K2_approx.clone();
        } else _degeneracy.dynamicCast<FundamentalDegeneracy>()->getApproximatedIntrinsics(vsac_output.K1, vsac_output.K2);
    }
    return true;
}
}}
