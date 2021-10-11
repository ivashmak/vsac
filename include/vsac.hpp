#ifndef VSAC_VSAC_HPP
#define VSAC_VSAC_HPP

namespace vsac {
enum SamplingMethod { SAMPLING_UNIFORM, SAMPLING_PROGRESSIVE_NAPSAC, SAMPLING_NAPSAC,
    SAMPLING_PROSAC };
enum LocalOptimMethod {LOCAL_OPTIM_NULL, LOCAL_OPTIM_INNER_LO, LOCAL_OPTIM_INNER_AND_ITER_LO,
    LOCAL_OPTIM_GC, LOCAL_OPTIM_SIGMA};
enum ScoreMethod {SCORE_METHOD_RANSAC, SCORE_METHOD_MSAC, SCORE_METHOD_MAGSAC, SCORE_METHOD_LMEDS};
enum NeighborSearchMethod { NEIGH_FLANN_KNN, NEIGH_GRID, NEIGH_FLANN_RADIUS };
enum EstimationMethod { Homography, Fundamental, Fundamental8, Essential, Affine, P3P, P6P};
enum VerificationMethod { NullVerifier, SprtVerifier, ASPRT};
enum PolishingMethod { NonePolisher, LSQPolisher, MAGSAC, CovPolisher};
enum ErrorMetric { SAMPSON_ERR, SGD_ERR, SYMM_REPR_ERR, FORW_REPR_ERR, RERPOJ};
enum MethodSolver { GEM_SOLVER, SVD_SOLVER };

/*
* Corrects image points with respect to H. Use good_point_mask to correct only good points.
* Output is corrected points of the same size. If point is not good then it is zero row in matrix
*/
bool getCorrectedPointsHomography (const cv::Mat &points1, const cv::Mat &points2, cv::Mat &corr_points1, cv::Mat &corr_points2, const cv::Mat &H, const std::vector<bool> &good_point_mask);

/*
* Corrects points using Lindstrom method, s.t. x'^T F x = 0
*/
void triangulatePointsLindstrom (const cv::Mat &F, const cv::Mat &points1, const cv::Mat &points2, cv::Mat &points1_corr, cv::Mat &points2_corr, const std::vector<bool> &good_point_mask);

/*
* If K1 and K2 are provided then algorithm output 3D points and correct pose pair (R, t) from essential matrix.
*/
void triangulatePointsLindstrom (const cv::Mat &F, const cv::Mat &points1, const cv::Mat &points2, cv::Mat &points1_corr, cv::Mat &points2_corr, const cv::Mat &K1, const cv::Mat &K2, cv::Mat &points3D, cv::Mat &R, cv::Mat &t, const std::vector<bool> &good_point_mask);

enum MODEL_CONFIDENCE {RANDOM, NON_RANDOM, UNKNOWN};
class Output {
private:
    std::vector<int> inliers;
public:
    cv::Mat model, K1, K2;
    // vector of number_inliers size
    // vector of points size, true if inlier, false - outlier
    std::vector<bool> inliers_mask;
    // vector of points size, value of i-th index corresponds to error of i-th point if i is inlier.
    std::vector<float> residuals;
    int number_inliers=0, number_iterations=0;
    MODEL_CONFIDENCE confidence=UNKNOWN;// not decided

    virtual ~Output() = default;
    Output () {}
    Output (const cv::Mat &model_, const std::vector<bool> &inliers_mask_, int number_inliers_,
            int number_iterations_, MODEL_CONFIDENCE conf, const std::vector<float> &errors_) {
        model_.copyTo(model);
        inliers_mask = inliers_mask_;
        number_inliers = number_inliers_;
        number_iterations = number_iterations_;
        confidence = conf;
        residuals = errors_;
    }

    // Return inliers' indices of size  = number of inliers
    const std::vector<int> &getInliers() {
        if (inliers.empty()) {
            inliers.reserve(number_inliers);
            int pt_cnt = 0;
            for (bool is_inlier : inliers_mask) {
                if (is_inlier)
                    inliers.emplace_back(pt_cnt);
                pt_cnt++;
            }
        }
        return inliers;
    }
};

class Params {
private:
    // main parameters:
    double threshold, confidence;
    int sample_size, max_iterations;

    EstimationMethod estimator;
    SamplingMethod sampler;
    ScoreMethod score;

    // for neighborhood graph
    int k_nearest_neighbors = 8;//, flann_search_params = 5, num_kd_trees = 1; // for FLANN
    int cell_size = 50; // pixels, for grid neighbors searching
    int radius = 30; // pixels, for radius-search neighborhood graph
    NeighborSearchMethod neighborsType = NeighborSearchMethod::NEIGH_GRID;

    // Local Optimization parameters
    LocalOptimMethod lo = LocalOptimMethod ::LOCAL_OPTIM_INNER_LO;
    int lo_sample_size=12, lo_inner_iterations=20, lo_iterative_iterations=8,
            lo_thr_multiplier=10, lo_iter_sample_size = 30;

    // Graph cut parameters
    const double spatial_coherence_term = 0.975;

    // apply polisher for final RANSAC model
    PolishingMethod polisher = PolishingMethod ::CovPolisher;

    // preemptive verification test
    VerificationMethod verifier = VerificationMethod ::ASPRT;
    const int max_hypothesis_test_before_verification = 30;

    // sprt parameters
    // lower bound estimate is 2% of inliers
    // model estimation to verification time = ratio of time needed to estimate model
    // to verification of one point wrt the model
    double sprt_eps = 0.02, sprt_delta = 0.008, avg_num_models, model_est_to_ver_time;

    // estimator error
    ErrorMetric est_error;

    // progressive napsac
    double relax_coef = 0.1;
    // for building neighborhood graphs
    const std::vector<int> grid_cell_number = {10, 5, 2};

    //for final least squares polisher
    int final_lsq_iters = 7;

    bool need_mask = true, // do we need inlier mask in the end
        is_parallel = false, // use parallel RANSAC
        enforce_singular_vals = true, // enforce singular values for F and E estimation
        is_nonrand_test = false, // is test for the final model non-randomness
        IS_QUASI_SAMPLING = false; // is quasi-pseudo-random sampling

    // state of pseudo-random number generator
    int random_generator_state = 0;

    // number of iterations to be done by RANSAC before applying local optimization
    const int max_iters_before_LO = 100;

    // solver for a null-space extraction
    MethodSolver r_solver = GEM_SOLVER;

    // number of iterations of plane-and-parallax in DEGENSAC^+
    int plane_and_parallax_max_iters = 100;

    // magsac parameters:
    int DoF = 2;
    double sigma_quantile = 3.04, upper_incomplete_of_sigma_quantile = 0.00419,
            lower_incomplete_of_sigma_quantile = 0.8629, C = 0.5, maximum_thr = 7.5;

    double k_mlesac = 2.25; // parameter for MLESAC model evaluation
    cv::Size img1_size, img2_size; // size of images
public:
    Params (EstimationMethod estimator_, double threshold_=1.0, double confidence_=0.95, int max_iterations_=5000,
            SamplingMethod sampler_=SAMPLING_UNIFORM, ScoreMethod score_ =ScoreMethod::SCORE_METHOD_MSAC) :
           estimator(estimator_), threshold(threshold_), confidence(confidence_), max_iterations(max_iterations_), sampler(sampler_), score(score_) {
        switch (estimator_) {
            case (EstimationMethod::Affine):
                avg_num_models = 1; model_est_to_ver_time = 50;
                sample_size = 3; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Homography):
                avg_num_models = 0.8; model_est_to_ver_time = 200;
                sample_size = 4; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Fundamental):
                DoF = 4; C = 0.25; sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.003657; lower_incomplete_of_sigma_quantile = 1.3012;
                maximum_thr = 2.5;
                avg_num_models = 1.5; model_est_to_ver_time = 200;
                sample_size = 7; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Fundamental8):
                avg_num_models = 1; model_est_to_ver_time = 100; maximum_thr = 2.5;
                sample_size = 8; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Essential):
                DoF = 4; C = 0.25; sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.003657; lower_incomplete_of_sigma_quantile = 1.3012;
                avg_num_models = 3.93; model_est_to_ver_time = 1000; maximum_thr = 2;
                sample_size = 5; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::P3P):
                avg_num_models = 1.38; model_est_to_ver_time = 800;
                sample_size = 3; est_error = ErrorMetric ::RERPOJ; break;
            case (EstimationMethod::P6P):
                avg_num_models = 1; model_est_to_ver_time = 300;
                sample_size = 6; est_error = ErrorMetric ::RERPOJ; break;
            default: CV_Error(cv::Error::StsNotImplemented, "Estimator has not implemented yet!");
        }

        // for PnP problem we can use only KNN graph
        if (estimator_ == EstimationMethod::P3P || estimator_ == EstimationMethod::P6P) {
            polisher = LSQPolisher;
            neighborsType = NeighborSearchMethod::NEIGH_FLANN_KNN;
            k_nearest_neighbors = 2;
        }
    }

    // setters
    void setNonRandomnessTest (bool set) { is_nonrand_test = set; }
    void setQuasiSampling (bool is_quasi) { IS_QUASI_SAMPLING = is_quasi; }
    void setVerifier (VerificationMethod verifier_) { verifier = verifier_; }
    void setPolisher (PolishingMethod polisher_) { polisher = polisher_; }
    void setParallel (bool is_parallel_) { is_parallel = is_parallel_; }
    void setError (ErrorMetric error_) { est_error = error_; }
    void setLocalOptimization (LocalOptimMethod lo_) { lo = lo_; }
    void setKNearestNeighhbors (int knn_) { k_nearest_neighbors = knn_; }
    void setNeighborsType (NeighborSearchMethod neighbors) { neighborsType = neighbors; }
    void setCellSize (int cell_size_) { cell_size = cell_size_; }
    void setLOIterations (int iters) { lo_inner_iterations = iters; }
    void setLOSampleSize (int lo_sample_size_) { lo_sample_size = lo_sample_size_; }
    void maskRequired (bool need_mask_) { need_mask = need_mask_; }
    void setPlaneAndParallaxIters (int iters) { plane_and_parallax_max_iters = iters; }
    void setImage1Size (cv::Size img_size_) {img1_size = img_size_; }
    void setImage2Size (cv::Size img_size_) {img2_size = img_size_; }
    void setImagesSize (cv::Size img1_size_, cv::Size img2_size_) {img1_size = img1_size_; img2_size = img2_size_; }
    void setRandomGeneratorState (int state) { random_generator_state = state; }
    void setKmlesac (double k) { k_mlesac = k; }
    void setRansacSolver (MethodSolver s) { r_solver = s; }

    // getters
    bool isNonRandomnessTest () const { return is_nonrand_test; }
    bool isQuasiSampling () const { return IS_QUASI_SAMPLING; }
    bool isEnforceRank () const { return enforce_singular_vals; }
    void setEnforceRank (bool enforce) { enforce_singular_vals = enforce; }
    void setFinalLSQiters (int iters) { final_lsq_iters = iters; }
    bool isMaskRequired () const { return need_mask; }
    NeighborSearchMethod getNeighborsSearch () const { return neighborsType; }
    int getKNN () const { return k_nearest_neighbors; }
    ErrorMetric getError () const { return est_error; }
    EstimationMethod getEstimator () const { return estimator; }
    int getSampleSize () const { return sample_size; }
    int getFinalLSQIterations () const { return final_lsq_iters; }
    int getDegreesOfFreedom () const { return DoF; }
    double getSigmaQuantile () const { return sigma_quantile; }
    double getUpperIncompleteOfSigmaQuantile () const {
        return upper_incomplete_of_sigma_quantile;
    }
    double getLowerIncompleteOfSigmaQuantile () const {
        return lower_incomplete_of_sigma_quantile;
    }
    double getC () const { return C; }
    double getKmlesac () const { return k_mlesac; }
    double getMaximumThreshold () const { return maximum_thr; }
    double getGraphCutSpatialCoherenceTerm () const { return spatial_coherence_term; }
    int getLOSampleSize () const { return lo_sample_size; }
    MethodSolver getRansacSolver () const { return r_solver; }
    PolishingMethod getFinalPolisher () const { return polisher; }
    int getLOThresholdMultiplier() const { return lo_thr_multiplier; }
    int getLOIterativeSampleSize() const { return lo_iter_sample_size; }
    int getLOIterativeMaxIters() const { return lo_iterative_iterations; }
    int getLOInnerMaxIters() const { return lo_inner_iterations; }
    int getPlaneAndParallaxIters () const { return plane_and_parallax_max_iters; }
    LocalOptimMethod getLO () const { return lo; }
    ScoreMethod getScore () const { return score; }
    int getMaxIters () const { return max_iterations; }
    double getConfidence () const { return confidence; }
    double getThreshold () const { return threshold; }
    VerificationMethod getVerifier () const { return verifier; }
    SamplingMethod getSampler () const { return sampler; }
    int getRandomGeneratorState () const { return random_generator_state; }
    int getMaxItersBeforeLO () const { return max_iters_before_LO; }
    double getSPRTdelta () const { return sprt_delta; }
    double getSPRTepsilon () const { return sprt_eps; }
    double getSPRTavgNumModels () const { return avg_num_models; }
    int getCellSize () const { return cell_size; }
    int getGraphRadius() const { return radius; }
    double getTimeForModelEstimation () const { return model_est_to_ver_time; }
    double getRelaxCoef () const { return relax_coef; }
    const std::vector<int> &getGridCellNumber () const { return grid_cell_number; }
    const cv::Size &getImage1Size () const { return img1_size; }
    const cv::Size &getImage2Size () const { return img2_size; }
    bool isParallel () const { return is_parallel; }
    bool isFundamental () const {
        return estimator == EstimationMethod ::Fundamental ||
               estimator == EstimationMethod ::Fundamental8;
    }
    bool isHomography () const { return estimator == EstimationMethod ::Homography; }
    bool isEssential () const { return estimator == EstimationMethod ::Essential; }
    bool isPnP() const {
        return estimator == EstimationMethod ::P3P || estimator == EstimationMethod ::P6P;
    }
};
bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
               Output &output, cv::InputArray K1_, cv::InputArray K2_,
               cv::InputArray dist_coeff1, cv::InputArray dist_coeff2);
bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
   cv::InputArray points1_val, cv::InputArray points2_val,
   Output &output, cv::InputArray K1_, cv::InputArray K2_,
   cv::InputArray dist_coeff1, cv::InputArray dist_coeff2);
}


#endif //VSAC_VSAC_HPP
