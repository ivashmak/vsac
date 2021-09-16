#include "precomp.hpp"
#include <opencv2/imgproc/detail/gcgraph.hpp>

namespace cv { namespace vsac {
class GraphCutImpl : public GraphCut {
protected:
    const Ptr<NeighborhoodGraph> neighborhood_graph;
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<RandomGenerator> lo_sampler;
    const Ptr<Error> error;

    int gc_sample_size, lo_inner_iterations, points_size;
    double spatial_coherence, sqr_trunc_thr, one_minus_lambda;

    std::vector<int> labeling_inliers;
    std::vector<double> energies, weights;
    std::vector<bool> used_edges;
    std::vector<Mat> gc_models;

    Ptr<Termination> termination;
    int num_lo_optimizations = 0, current_ransac_iter = 0;
public:
    void setCurrentRANSACiter (int ransac_iter) override { current_ransac_iter = ransac_iter; }

    // In lo_sampler_ the sample size should be set and be equal gc_sample_size_
    GraphCutImpl (const Ptr<Estimator> &estimator_, const Ptr<Error> &error_, const Ptr<Quality> &quality_,
              const Ptr<NeighborhoodGraph> &neighborhood_graph_, const Ptr<RandomGenerator> &lo_sampler_,
              double threshold_, double spatial_coherence_term, int gc_inner_iteration_number_, Ptr<Termination> termination_) :
              neighborhood_graph (neighborhood_graph_), estimator (estimator_), quality (quality_),
              lo_sampler (lo_sampler_), error (error_), termination(termination_) {

        points_size = quality_->getPointsSize();
        spatial_coherence = spatial_coherence_term;
        sqr_trunc_thr =  threshold_ * 2.25; // threshold is already squared
        gc_sample_size = lo_sampler_->getSubsetSize();
        lo_inner_iterations = gc_inner_iteration_number_;
        one_minus_lambda = 1.0 - spatial_coherence;

        energies = std::vector<double>(points_size);
        labeling_inliers = std::vector<int>(points_size);
        used_edges = std::vector<bool>(points_size*points_size);
        gc_models = std::vector<Mat> (estimator->getMaxNumSolutionsNonMinimal());
    }

    bool refineModel (const Mat &best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < estimator->getNonMinimalSampleSize())
            return false;

        // improve best model by non minimal estimation
        new_model_score = Score(); // set score to inf (worst case)
        best_model.copyTo(new_model);

        bool is_best_model_updated = true;
        while (is_best_model_updated) {
            is_best_model_updated = false;

            // Build graph problem. Apply graph cut to G
            int labeling_inliers_size = labeling(new_model);
            // std::cout << "labeling_inliers_size " << labeling_inliers_size << '\n';
            for (int iter = 0; iter < lo_inner_iterations; iter++) {
                // sample to generate min (|I_7m|, |I|)
                int num_of_estimated_models;
                if (labeling_inliers_size > gc_sample_size) {
                    // generate random subset in range <0; |I|>
                    num_of_estimated_models = estimator->estimateModelNonMinimalSample
                            (lo_sampler->generateUniqueRandomSubset(labeling_inliers,
                                   labeling_inliers_size), gc_sample_size, gc_models, weights);
                } else {
                    if (iter > 0) break; // break inliers are not updated
                    num_of_estimated_models = estimator->estimateModelNonMinimalSample
                            (labeling_inliers, labeling_inliers_size, gc_models, weights);
                }
                for (int model_idx = 0; model_idx < num_of_estimated_models; model_idx++) {
                    const Score gc_temp_score = quality->getScore(gc_models[model_idx]);
                    // store the best model from estimated models
                    if (gc_temp_score.isBetter(new_model_score)) {
                        is_best_model_updated = true;
                        new_model_score = gc_temp_score;
                        gc_models[model_idx].copyTo(new_model);
                    }
                }

                if (termination != nullptr && is_best_model_updated && current_ransac_iter > termination->update(best_model, best_model_score.inlier_number)) {
                    is_best_model_updated = false; // to break outer loop
                }

            } // end of inner GC local optimization
        } // end of while loop
        return true;
    }

private:
    // find inliers using graph cut algorithm.
    int labeling (const Mat& model) {
        const auto &errors = error->getErrors(model);
        detail::GCGraph<double> graph;

        for (int pt = 0; pt < points_size; pt++)
            graph.addVtx();

        // The distance and energy for each point
        double tmp_squared_distance, energy;

        // Estimate the vertex capacities
        for (int pt = 0; pt < points_size; pt++) {
            tmp_squared_distance = errors[pt];
            if (std::isnan(tmp_squared_distance))
                tmp_squared_distance = std::numeric_limits<float>::max();
            energy = tmp_squared_distance / sqr_trunc_thr; // Truncated quadratic cost

            if (tmp_squared_distance <= sqr_trunc_thr)
                graph.addTermWeights(pt, 0, one_minus_lambda * (1 - energy));
            else
                graph.addTermWeights(pt, one_minus_lambda * energy, 0);

            energies[pt] = energy > 1 ? 1 : energy;
        }

        std::fill(used_edges.begin(), used_edges.end(), false);

        bool has_edges = false;
        // Iterate through all points and set their edges
        for (int point_idx = 0; point_idx < points_size; ++point_idx) {
            energy = energies[point_idx];

            // Iterate through  all neighbors
            for (int actual_neighbor_idx : neighborhood_graph->getNeighbors(point_idx)) {
                if (actual_neighbor_idx == point_idx ||
                    used_edges[actual_neighbor_idx*points_size + point_idx] ||
                    used_edges[point_idx*points_size + actual_neighbor_idx])
                    continue;

                used_edges[actual_neighbor_idx*points_size + point_idx] = true;
                used_edges[point_idx*points_size + actual_neighbor_idx] = true;

                double a = (0.5 * (energy + energies[actual_neighbor_idx])) * spatial_coherence,
                       b = spatial_coherence, c = spatial_coherence, d = 0;
                graph.addTermWeights(point_idx, d, a);
                b -= a;
                if (b + c < 0)
                    continue; // invalid regularity
                if (b < 0) {
                    graph.addTermWeights(point_idx, 0, b);
                    graph.addTermWeights(actual_neighbor_idx, 0, -b);
                    graph.addEdges(point_idx, actual_neighbor_idx, 0, b + c);
                } else if (c < 0) {
                    graph.addTermWeights(point_idx, 0, -c);
                    graph.addTermWeights(actual_neighbor_idx, 0, c);
                    graph.addEdges(point_idx, actual_neighbor_idx, b + c, 0);
                } else
                    graph.addEdges(point_idx, actual_neighbor_idx, b, c);
                has_edges = true;
            }
        }
        if (! has_edges)
            return quality->getInliers(model, labeling_inliers);
        graph.maxFlow();

        int inlier_number = 0;
        for (int pt = 0; pt < points_size; pt++)
            if (! graph.inSourceSegment(pt)) // check for sink
                labeling_inliers[inlier_number++] = pt;
        return inlier_number;
    }
    int getNumLOoptimizations () const override { return num_lo_optimizations; }
};
Ptr<GraphCut> GraphCut::create(const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
       const Ptr<Quality> &quality_, const Ptr<NeighborhoodGraph> &neighborhood_graph_,
       const Ptr<RandomGenerator> &lo_sampler_, double threshold_,
       double spatial_coherence_term, int gc_inner_iteration_number, Ptr<Termination> termination_) {
    return makePtr<GraphCutImpl>(estimator_, error_, quality_, neighborhood_graph_, lo_sampler_,
        threshold_, spatial_coherence_term, gc_inner_iteration_number, termination_);
}

// http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
class InnerIterativeLocalOptimizationImpl : public InnerIterativeLocalOptimization {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<RandomGenerator> lo_sampler;
    Ptr<RandomGenerator> lo_iter_sampler;

    std::vector<Mat> lo_models, lo_iter_models;

    std::vector<int> inliers_of_best_model, virtual_inliers;
    int lo_inner_max_iterations, lo_iter_max_iterations, lo_sample_size, lo_iter_sample_size;

    bool is_iterative;

    double threshold, new_threshold, threshold_step;
    std::vector<double> weights;
public:
    InnerIterativeLocalOptimizationImpl (const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
         const Ptr<RandomGenerator> &lo_sampler_, int pts_size,
         double threshold_, bool is_iterative_, int lo_iter_sample_size_,
         int lo_inner_iterations_=10, int lo_iter_max_iterations_=5,
         double threshold_multiplier_=4)
        : estimator (estimator_), quality (quality_), lo_sampler (lo_sampler_)
        , lo_iter_sample_size(0), new_threshold(0), threshold_step(0) {
        lo_inner_max_iterations = lo_inner_iterations_;
        lo_iter_max_iterations = lo_iter_max_iterations_;

        threshold = threshold_;
        lo_sample_size = lo_sampler->getSubsetSize();
        is_iterative = is_iterative_;
        if (is_iterative) {
            lo_iter_sample_size = lo_iter_sample_size_;
            lo_iter_sampler = UniformRandomGenerator::create(0/*state*/, pts_size, lo_iter_sample_size_);
            lo_iter_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
            virtual_inliers = std::vector<int>(pts_size);
            new_threshold = threshold_multiplier_ * threshold;
            // reduce multiplier threshold K·θ by this number in each iteration.
            // In the last iteration there be original threshold θ.
            threshold_step = (new_threshold - threshold) / lo_iter_max_iterations_;
        }
        lo_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());

        // Allocate max memory to avoid reallocation
        inliers_of_best_model = std::vector<int>(pts_size);
    }

    /*
     * Implementation of Locally Optimized Ransac
     * Inner + Iterative
     */
    bool refineModel (const Mat &so_far_the_best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < estimator->getNonMinimalSampleSize())
            return false;

        so_far_the_best_model.copyTo(new_model);
        new_model_score = best_model_score;
        // get inliers from so far the best model.
        int num_inliers_of_best_model = quality->getInliers(so_far_the_best_model,
                                                           inliers_of_best_model);

        // Inner Local Optimization Ransac.
        for (int iters = 0; iters < lo_inner_max_iterations; iters++) {
            int num_estimated_models;
            // Generate sample of lo_sample_size from inliers from the best model.
            if (num_inliers_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sampler->generateUniqueRandomSubset(inliers_of_best_model,
                                num_inliers_of_best_model), lo_sample_size, lo_models, weights);
            } else {
                // if model was not updated in first iteration, so break.
                if (iters > 0) break;
                // if inliers are less than limited number of sample then take all for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                    (inliers_of_best_model, num_inliers_of_best_model, lo_models, weights);
            }

            //////// Choose the best lo_model from estimated lo_models.
            for (int model_idx = 0; model_idx < num_estimated_models; model_idx++) {
                const Score temp_score = quality->getScore(lo_models[model_idx]);
                if (temp_score.isBetter(new_model_score)) {
                    new_model_score = temp_score;
                    lo_models[model_idx].copyTo(new_model);
                }
            }

            if (is_iterative) {
                double lo_threshold = new_threshold;
                // get max virtual inliers. Note that they are nor real inliers,
                // because we got them with bigger threshold.
                int virtual_inliers_size = quality->getInliers
                        (new_model, virtual_inliers, lo_threshold);

                Mat lo_iter_model;
                Score lo_iter_score = Score(); // set worst case
                for (int iterations = 0; iterations < lo_iter_max_iterations; iterations++) {
                    lo_threshold -= threshold_step;

                    if (virtual_inliers_size > lo_iter_sample_size) {
                        // if there are more inliers than limit for sample size then generate at random
                        // sample from LO model.
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (lo_iter_sampler->generateUniqueRandomSubset (virtual_inliers,
                            virtual_inliers_size), lo_iter_sample_size, lo_iter_models, weights);
                    } else {
                        // break if failed, very low probability that it will not fail in next iterations
                        // estimate model with all virtual inliers
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (virtual_inliers, virtual_inliers_size, lo_iter_models, weights);
                    }
                    if (num_estimated_models == 0) break;

                    // Get score and update virtual inliers with current threshold
                    ////// Choose the best lo_iter_model from estimated lo_iter_models.
                    lo_iter_models[0].copyTo(lo_iter_model);
                    lo_iter_score = quality->getScore(lo_iter_model);
                    for (int model_idx = 1; model_idx < num_estimated_models; model_idx++) {
                        const Score temp_score = quality->getScore(lo_iter_models[model_idx]);
                        if (temp_score.isBetter(lo_iter_score)) {
                            lo_iter_score = temp_score;
                            lo_iter_models[model_idx].copyTo(lo_iter_model);
                        }
                    }

                    if (iterations != lo_iter_max_iterations-1)
                        virtual_inliers_size = quality->getInliers(lo_iter_model, virtual_inliers, lo_threshold);
                }

                if (lo_iter_score.isBetter(new_model_score)) {
                    new_model_score = lo_iter_score;
                    lo_iter_model.copyTo(new_model);
                }
            }

            if (num_inliers_of_best_model < new_model_score.inlier_number && iters != lo_inner_max_iterations-1)
                num_inliers_of_best_model = quality->getInliers (new_model, inliers_of_best_model);
        }
        return true;
    }
};
Ptr<InnerIterativeLocalOptimization> InnerIterativeLocalOptimization::create
(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
       const Ptr<RandomGenerator> &lo_sampler_, int pts_size,
       double threshold_, bool is_iterative_, int lo_iter_sample_size_,
       int lo_inner_iterations_, int lo_iter_max_iterations_,
       double threshold_multiplier_) {
    return makePtr<InnerIterativeLocalOptimizationImpl>(estimator_, quality_, lo_sampler_,
            pts_size, threshold_, is_iterative_, lo_iter_sample_size_,
            lo_inner_iterations_, lo_iter_max_iterations_, threshold_multiplier_);
}

class SimpleLocalOptimizationImpl : public SimpleLocalOptimization {
private:
    const Ptr<Degeneracy> degen;
    const Ptr<Quality> quality;
    const Ptr<Error> error;
    const Ptr<Estimator> estimator;
    const Ptr<Termination> termination;
    const Ptr<RandomGenerator> random_generator;
    int points_size, max_lo_iters, non_min_sample_size, max_iters_ransac, current_ransac_iter;
    std::vector<double> weights;
    std::vector<int> inliers;
    std::vector<cv::Mat> models;
    double inlier_threshold_sqr;

    int num_lo_optimizations = 0;
public:
    SimpleLocalOptimizationImpl (const Ptr<Degeneracy> &degen_, const Ptr<Quality> &quality_, const Ptr<Estimator> &estimator_,
            const Ptr<Termination> termination_, const Ptr<RandomGenerator> &random_gen, int max_lo_iters_, double inlier_threshold_sqr_) :
            degen(degen_), quality(quality_), error(quality_->getErrorFnc()), estimator(estimator_), termination(termination_), random_generator(random_gen) {
        max_lo_iters = max_lo_iters_;
        non_min_sample_size = random_generator->getSubsetSize();
        max_iters_ransac = INT_MAX;
        current_ransac_iter = 0;
        inliers = std::vector<int>(quality_->getPointsSize());
        models = std::vector<cv::Mat>(estimator_->getMaxNumSolutionsNonMinimal());
        points_size = quality_->getPointsSize();
        inlier_threshold_sqr = inlier_threshold_sqr_;
    }
    void setCurrentRANSACiter (int ransac_iter) override { current_ransac_iter = ransac_iter; }
    int getMaxIterations () const override {
        return max_iters_ransac;
    }
    int getNumLOoptimizations () const override { return num_lo_optimizations; }
    bool refineModel (const Mat &best_model, const Score &best_model_score, Mat &new_model, Score &new_model_score) override {
        new_model_score = best_model_score;
        best_model.copyTo(new_model);

        int num_inliers = Quality::getInliers(error, best_model, inliers, inlier_threshold_sqr);
        // std::cout << "init num inliers " << num_inliers << '\n';
        Ptr<UniformRandomGenerator> random_generator2;
        if (num_inliers <= non_min_sample_size) {
            const int new_sample_size = (int)(0.6*num_inliers);
            if (new_sample_size <= estimator->getNonMinimalSampleSize())
                return false;
            random_generator2 = UniformRandomGenerator::create(0, quality->getPointsSize(), new_sample_size);
        }

        // error_fnc->setModelParameters(best_model);
        for (int iter = 0; iter < max_lo_iters; iter++) {
            int num_models;
            if (num_inliers <= non_min_sample_size) {
                // num_models = estimator->estimateModelNonMinimalSample(inliers, num_inliers, models, weights);
                num_models = estimator->estimateModelNonMinimalSample(random_generator2->generateUniqueRandomSubset(inliers, num_inliers),
                        random_generator2->getSubsetSize(), models, weights);
            } else {
                num_models = estimator->estimateModelNonMinimalSample(random_generator->generateUniqueRandomSubset(inliers, num_inliers), non_min_sample_size, models, weights);
            }

            for (int m = 0; m < num_models; m++) {
                const auto score = quality->getScore(models[m]);
                if (score.isBetter(new_model_score)) {
                    new_model_score = score;
                   // std::cout << "LO update best, new MLESAC score " << score.score << " best " << best_model_score.score << "\n";
                    models[m].copyTo(new_model);
                    num_inliers = Quality::getInliers(error, new_model, inliers, inlier_threshold_sqr);
                    if (termination != nullptr && current_ransac_iter > termination->update(new_model, new_model_score.inlier_number))
                        break;
                    if (num_inliers <= non_min_sample_size) {
                        const int new_sample_size = (int)(0.6*num_inliers);
                        if (new_sample_size <= 7)
                            return true;
                        random_generator2 = UniformRandomGenerator::create(0, quality->getPointsSize(), new_sample_size);
                    }
                }
            }
        }
        return true;
    }
};
Ptr<SimpleLocalOptimization> SimpleLocalOptimization::create (const Ptr<Degeneracy> &degen, const Ptr<Quality> &quality_,
        const Ptr<Estimator> &estimator_, const Ptr<Termination> termination_, const Ptr<RandomGenerator> &random_gen, int max_lo_iters_, double inlier_thr_sqr) {
    return makePtr<SimpleLocalOptimizationImpl> (degen, quality_, estimator_, termination_, random_gen, max_lo_iters_, inlier_thr_sqr);
}


class SigmaConsensusImpl : public SigmaConsensus {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<Error> error;
    const Ptr<ModelVerifier> verifier;
    const Ptr<GammaValues> gamma_generator;
    // The degrees of freedom of the data from which the model is estimated.
    // E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
    const int degrees_of_freedom;
    // A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
    const double k;
    // Calculating (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double dof_minus_one_per_two;
    const double C;
    // The size of a minimal sample used for the estimation
    const int sample_size;
    // Calculating 2^(DoF - 1) which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double two_ad_dof;
    // Calculating C * 2^(DoF - 1) which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double C_times_two_ad_dof;
    // Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double squared_sigma_max_2, one_over_sigma;
    // Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
    const double gamma_k;
    // Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double max_sigma_sqr;
    const int points_size, number_of_irwls_iters;
    const double maximum_threshold, max_sigma;

    std::vector<double> sqr_residuals, sigma_weights;
    std::vector<int> sqr_residuals_idxs;
    // Models fit by weighted least-squares fitting
    std::vector<Mat> sigma_models;
    // Points used in the weighted least-squares fitting
    std::vector<int> sigma_inliers;
    // Weights used in the the weighted least-squares fitting
    int max_lo_sample_size, stored_gamma_number_min1;
    double scale_of_stored_gammas;
    RNG rng;
    const std::vector<double> &stored_gamma_values;

    Ptr<Termination> termination;
    int num_lo_optimizations = 0, current_ransac_iter = 0;
public:
    void setCurrentRANSACiter (int ransac_iter) override { current_ransac_iter = ransac_iter; }

    SigmaConsensusImpl (const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
        const Ptr<Quality> &quality_, const Ptr<ModelVerifier> &verifier_,
        const Ptr<GammaValues> &gamma_generator_,
        int max_lo_sample_size_, int number_of_irwls_iters_, int DoF,
        double sigma_quantile, double upper_incomplete_of_sigma_quantile, double C_,
        double maximum_thr, Ptr<Termination> termination_) : estimator (estimator_), quality(quality_),
          error (error_), verifier(verifier_), gamma_generator(gamma_generator_),
          degrees_of_freedom(DoF), k (sigma_quantile), C(C_),
          sample_size(estimator_->getMinimalSampleSize()),
          gamma_k (upper_incomplete_of_sigma_quantile), points_size (quality_->getPointsSize()),
          number_of_irwls_iters (number_of_irwls_iters_),
          maximum_threshold(maximum_thr), max_sigma (maximum_thr),
          stored_gamma_values (gamma_generator_->getGammaValues()), termination(termination_) {

        dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
        two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
        C_times_two_ad_dof = C * two_ad_dof;
        // Calculate 2 * \sigma_{max}^2 a priori
        squared_sigma_max_2 = max_sigma * max_sigma * 2.0;
        // Divide C * 2^(DoF - 1) by \sigma_{max} a priori
        one_over_sigma = C_times_two_ad_dof / max_sigma;
        max_sigma_sqr = squared_sigma_max_2 * 0.5;
        sqr_residuals = std::vector<double>(points_size);
        sqr_residuals_idxs = std::vector<int>(points_size);
        sigma_inliers = std::vector<int>(points_size);
        max_lo_sample_size = max_lo_sample_size_;
        sigma_weights = std::vector<double>(points_size);
        sigma_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
        stored_gamma_number_min1 = gamma_generator->getTableSize()-1;
        scale_of_stored_gammas = gamma_generator->getScaleOfGammaValues();
    }

    // https://github.com/danini/magsac
    bool refineModel (const Mat &in_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        int residual_cnt = 0;
        error->setModelParameters(in_model);
        for (int point_idx = 0; point_idx < points_size; ++point_idx) {
            const double sqr_residual = error->getError(point_idx);
            if (sqr_residual < max_sigma_sqr) {
                // Store the residual of the current point and its index
                sqr_residuals[residual_cnt] = sqr_residual;
                sqr_residuals_idxs[residual_cnt++] = point_idx;
            }

            if (residual_cnt + points_size - point_idx < best_model_score.inlier_number)
                return false;
        }
        if (residual_cnt <= sample_size)
            return false;
        in_model.copyTo(new_model);
        new_model_score = Score();

        // Do the iteratively re-weighted least squares fitting
        bool is_updated = false;
        for (int iterations = 0; iterations < number_of_irwls_iters; iterations++) {
            int sigma_inliers_cnt = 0;
            // If the current iteration is not the first, the set of possibly inliers
            // (i.e., points closer than the maximum threshold) have to be recalculated.
            if (iterations > 0) {
                // error->setModelParameters(polished_model);
                error->setModelParameters(new_model);
                // Remove everything from the residual vector
                residual_cnt = 0;

                // Collect the points which are closer than the maximum threshold
                for (int point_idx = 0; point_idx < points_size; ++point_idx) {
                    // Calculate the residual of the current point
                    const double sqr_residual = error->getError(point_idx);
                    if (sqr_residual < max_sigma_sqr) {
                        // Store the residual of the current point and its index
                        sqr_residuals[residual_cnt] = sqr_residual;
                        sqr_residuals_idxs[residual_cnt++] = point_idx;
                    }
                }
                sigma_inliers_cnt = 0;
            }

            // Calculate the weight of each point
            for (int i = 0; i < residual_cnt; i++) {
                // Get the position of the gamma value in the lookup table
                int x = (int)round(scale_of_stored_gammas * sqr_residuals[i]
                        / squared_sigma_max_2);

                // If the sought gamma value is not stored in the lookup, return the closest element
                if (x >= stored_gamma_number_min1 || x < 0 /*overflow*/) // actual number of gamma values is 1 more, so >=
                    x  = stored_gamma_number_min1;

                sigma_inliers[sigma_inliers_cnt] = sqr_residuals_idxs[i]; // store index of point for LSQ
                sigma_weights[sigma_inliers_cnt++] = one_over_sigma * (stored_gamma_values[x] - gamma_k);
            }

            // random shuffle sigma inliers
            if (sigma_inliers_cnt > max_lo_sample_size)
                for (int i = sigma_inliers_cnt-1; i > 0; i--) {
                    const int idx = rng.uniform(0, i+1);
                    std::swap(sigma_inliers[i], sigma_inliers[idx]);
                    std::swap(sigma_weights[i], sigma_weights[idx]);
                }
            else if (iterations > 0 && !is_updated) break;
            const int num_est_models = estimator->estimateModelNonMinimalSample
                  (sigma_inliers, std::min(max_lo_sample_size, sigma_inliers_cnt),
                          sigma_models, sigma_weights);

            if (num_est_models == 0)
                break; // break iterations

            // Update the model parameters
            Mat polished_model = sigma_models[0];
            Score sigma_best_score = quality->getScore(polished_model);
            if (num_est_models > 1) {
                // find best over other models
                for (int m = 1; m < num_est_models; m++) {
                    const Score sc = quality->getScore(sigma_models[m]);
                    if (sc.isBetter(sigma_best_score)) {
                        polished_model = sigma_models[m];
                        sigma_best_score = sc;
                    }
                }
            }
            if (sigma_best_score.isBetter(new_model_score)) {
                new_model_score = sigma_best_score;
                polished_model.copyTo(new_model);
                is_updated = true;
                if (termination != nullptr && current_ransac_iter > termination->update(new_model, new_model_score.inlier_number))
                    break;
            } else is_updated = false;
        }

        return true;
    }
    int getNumLOoptimizations () const override { return num_lo_optimizations; }
};
Ptr<SigmaConsensus>
SigmaConsensus::create(const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
        const Ptr<Quality> &quality, const Ptr<ModelVerifier> &verifier_,
        const Ptr<GammaValues> &gamma_generator, int max_lo_sample_size, int number_of_irwls_iters_, int DoF,
        double sigma_quantile, double upper_incomplete_of_sigma_quantile, double C_,
        double maximum_thr, Ptr<Termination> termination_) {
    return makePtr<SigmaConsensusImpl>(estimator_, error_, quality, verifier_, gamma_generator,
            max_lo_sample_size, number_of_irwls_iters_, DoF, sigma_quantile,
            upper_incomplete_of_sigma_quantile, C_, maximum_thr, termination_);
}

/////////////////////////////////////////// FINAL MODEL POLISHER ////////////////////////
class LeastSquaresPolishingImpl : public LeastSquaresPolishing {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    int lsq_iterations;
    std::vector<int> inliers;
    std::vector<Mat> models;
    std::vector<double> weights;
public:
    LeastSquaresPolishingImpl(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
            int lsq_iterations_) :
            estimator(estimator_), quality(quality_) {
        lsq_iterations = lsq_iterations_;
        // allocate memory for inliers array and models
        inliers = std::vector<int>(quality_->getPointsSize());
        models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
    }

    bool polishSoFarTheBestModel(const Mat &model, const Score &best_model_score,
                                 Mat &new_model, Score &out_score) override {
        // get inliers from input model
        int inlier_number = quality->getInliers(model, inliers);
        if (inlier_number < estimator->getMinimalSampleSize())
            return false;

        out_score = Score(); // set the worst case

        // several all-inlier least-squares refines model better than only one but for
        // big amount of points may be too time-consuming.
        for (int lsq_iter = 0; lsq_iter < lsq_iterations; lsq_iter++) {
            bool model_updated = false;

            // estimate non minimal models with all inliers
            const int num_models = estimator->estimateModelNonMinimalSample(inliers,
                                                      inlier_number, models, weights);
            for (int model_idx = 0; model_idx < num_models; model_idx++) {
                const Score score = quality->getScore(models[model_idx]);
                if (best_model_score.isBetter(score))
                    continue;
                if (score.isBetter(out_score)) {
                    models[model_idx].copyTo(new_model);
                    out_score = score;
                    model_updated = true;
                }
            }

            if (!model_updated)
                // if model was not updated at the first iteration then return false
                // otherwise if all-inliers LSQ has not updated model then no sense
                // to do it again -> return true (model was updated before).
                return lsq_iter > 0;

            // if number of inliers doesn't increase more than 5% then break
            if (fabs(static_cast<double>(out_score.inlier_number) - static_cast<double>
                 (best_model_score.inlier_number)) / best_model_score.inlier_number < 0.05)
                return true;

            if (lsq_iter != lsq_iterations - 1)
                // if not the last LSQ normalization then get inliers for next normalization
                inlier_number = quality->getInliers(new_model, inliers);
        }
        return true;
    }
};
Ptr<LeastSquaresPolishing> LeastSquaresPolishing::create (const Ptr<Estimator> &estimator_,
         const Ptr<Quality> &quality_, int lsq_iterations_) {
    return makePtr<LeastSquaresPolishingImpl>(estimator_, quality_, lsq_iterations_);
}

class IterativePolisherImpl : public IterativePolisher {
private:
    const Ptr<Quality> quality;
    const Ptr<Estimator> estimator;
    Ptr<SimpleLocalOptimization> simple_lo;
    Ptr<Termination> termination;
public:
    IterativePolisherImpl (const Ptr<Quality> &quality_, const Ptr<Degeneracy> &degen, const Ptr<Estimator> &estimator_,
         const Ptr<RandomGenerator> &random_gen, int iters) : quality(quality_), estimator (estimator_) {
        // does not really matter here
        termination = StandardTerminationCriteria::create(0.99, quality_->getPointsSize(), estimator_->getMinimalSampleSize(), 100);
        simple_lo = SimpleLocalOptimization::create(degen, quality_, estimator_, termination, random_gen, iters, quality_->getThreshold());
        simple_lo->setCurrentRANSACiter(-1); // do full LO
    }
    bool polishSoFarTheBestModel(const Mat &model, const Score &best_model_score,
                                 Mat &new_model, Score &out_score) override {
        Score new_score;
        const bool success = simple_lo->refineModel(model, best_model_score, new_model, new_score);
        if (best_model_score.isBetter(new_score)) {
            // try all inlier LSQ
            std::vector<int> inliers(quality->getPointsSize());
            std::vector<Mat> models;
            std::vector<double> weights;
            const int num_inliers = quality->getInliers(model, inliers);
            const int num_models = estimator->estimateModelNonMinimalSample(inliers, num_inliers, models, weights);
            for (const auto &m : models) {
                const auto sc = quality->getScore(m);
                if (sc.isBetter(new_score)) {
                    new_score = sc;
                    m.copyTo(new_model);
                }
            }
            out_score = new_score;
            return new_score.isBetter(best_model_score);
        } else {
            out_score = new_score;
            return true;
        }
    }
};
Ptr<IterativePolisher> IterativePolisher::create (const Ptr<Quality> &quality_, const Ptr<Degeneracy> &degen, const Ptr<Estimator> &estimator_,
         const Ptr<RandomGenerator> &random_gen, int iters) {
    return makePtr<IterativePolisherImpl>(quality_, degen, estimator_, random_gen, iters);
}
}}
