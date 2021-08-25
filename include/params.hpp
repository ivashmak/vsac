#ifndef VSAC_PARAMS_HPP
#define VSAC_PARAMS_HPP


namespace vsac{
enum SamplingMethod { SAMPLING_UNIFORM, SAMPLING_PROGRESSIVE_NAPSAC, SAMPLING_NAPSAC,
    SAMPLING_PROSAC };
enum LocalOptimMethod {LOCAL_OPTIM_NULL, LOCAL_OPTIM_INNER_LO, LOCAL_OPTIM_INNER_AND_ITER_LO,
    LOCAL_OPTIM_GC, LOCAL_OPTIM_SIGMA};
enum ScoreMethod {SCORE_METHOD_RANSAC, SCORE_METHOD_MSAC, SCORE_METHOD_MAGSAC, SCORE_METHOD_LMEDS};
enum NeighborSearchMethod { NEIGH_FLANN_KNN, NEIGH_GRID, NEIGH_FLANN_RADIUS };

enum EstimationMethod { Homography, Fundamental, Fundamental8, Essential, Affine, P3P, P6P};
enum VerificationMethod { NullVerifier, SprtVerifier, ASPRT};
enum PolishingMethod { NonePolisher, LSQPolisher, MAGSAC, CovPolisher, IterativePolish};
enum ErrorMetric { SAMPSON_ERR, SGD_ERR, SYMM_REPR_ERR, FORW_REPR_ERR, RERPOJ};
enum MethodSolver { GEM_SOLVER, SVD_SOLVER };


bool getCorrectedPointsHomography (const cv::Mat &points, cv::Mat &corr_points, const cv::Mat &H);
bool getCorrectedPointsHomography (const cv::Mat &points, cv::Mat &corr_points, const cv::Mat &H, const std::vector<bool> &good_point_mask);
void triangulatePointsLindstrom (const cv::Mat &points, const cv::Mat &model, bool is_fundamental, cv::Mat &new_points, cv::Mat &points3D, cv::Mat &R1);
class Output {
private:
    cv::Mat model;
    // vector of number_inliers size
    std::vector<int> inliers;
    // vector of points size, true if inlier, false-outlier
    std::vector<bool> inliers_mask;
    // vector of points size, value of i-th index corresponds to error of i-th point if i is inlier.
    std::vector<float> errors;
    int number_inliers=0, number_iterations=0;
    bool is_random=false;// not decided
public:
    virtual ~Output() = default;
    Output () {}
    Output (const cv::Mat &model_, const std::vector<bool> &inliers_mask_, int number_inliers_,
                int number_iterations_, bool is_random_, const std::vector<float> &errors_) {
        model_.copyTo(model);
        inliers_mask = inliers_mask_;
        number_inliers = number_inliers_;
        number_iterations = number_iterations_;
        is_random = is_random_;
        errors = errors_;
    }

    // Return inliers' indiceso of size  = number of inliers
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

    // Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
    const std::vector<bool> &getInliersMask() const { return inliers_mask; }
    int getNumberOfInliers() const { return number_inliers; }
    int getNumberOfIterations() const { return number_iterations; }
    const cv::Mat &getModel() const { return model; }
    bool isModelRandom() const { return is_random; }
    const std::vector<float> &getPointResiduals () const { return errors; }
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
    int lo_sample_size=12, lo_inner_iterations=15, lo_iterative_iterations=8,
            lo_thr_multiplier=15, lo_iter_sample_size = 30;

    // Graph cut parameters
    const double spatial_coherence_term = 0.975;

    // apply polisher for final RANSAC model
    PolishingMethod polisher = PolishingMethod ::CovPolisher;

    // preemptive verification test
    VerificationMethod verifier = VerificationMethod ::ASPRT;
    const int max_hypothesis_test_before_verification = 30;

    // sprt parameters
    // lower bound estimate is 1% of inliers
    double sprt_eps = 0.02, sprt_delta = 0.008, avg_num_models, time_for_model_est;

    // estimator error
    ErrorMetric est_error;

    // progressive napsac
    double relax_coef = 0.1;
    // for building neighborhood graphs
    const std::vector<int> grid_cell_number = {10, 5, 2};

    //for final least squares polisher
    int final_lsq_iters = 5;

    bool need_mask = true, is_parallel = false, force_lo = false, enforce_singular_vals = true;
    bool is_nonrand_test = false, IS_QUASI_SAMPLING = false;
    int random_generator_state = 0;
    const int max_iters_before_LO = 100;

    MethodSolver r_solver = GEM_SOLVER;

    int plane_and_parallax_max_iters = 50;
    // magsac parameters:
    int DoF = 2;
    double sigma_quantile = 3.04, upper_incomplete_of_sigma_quantile = 0.00419,
            lower_incomplete_of_sigma_quantile = 0.8629, C = 0.5, maximum_thr = 7.5;
    double k_mlesac = 2.25;
    cv::Size img_size;

public:
    Params (double threshold_, EstimationMethod estimator_, SamplingMethod sampler_, double confidence_=0.95,
               int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::SCORE_METHOD_MSAC) {
        estimator = estimator_;
        sampler = sampler_;
        confidence = confidence_;
        max_iterations = max_iterations_;
        score = score_;

        switch (estimator_) {
            // time for model estimation is basically a ratio of time need to estimate a model to
            // time needed to verify if a point is consistent with this model
            case (EstimationMethod::Affine):
                avg_num_models = 1; time_for_model_est = 50;
                sample_size = 3; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Homography):
                avg_num_models = 0.8; time_for_model_est = 200;
                sample_size = 4; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Fundamental):
                DoF = 4; C = 0.25; sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.003657; lower_incomplete_of_sigma_quantile = 1.3012;
                maximum_thr = 2.5;
                avg_num_models = 1.5; time_for_model_est = 200;
                sample_size = 7; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Fundamental8):
                avg_num_models = 1; time_for_model_est = 100; maximum_thr = 5;
                sample_size = 8; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Essential):
                DoF = 4; C = 0.25; sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.003657; lower_incomplete_of_sigma_quantile = 1.3012;
                avg_num_models = 3.93; time_for_model_est = 1000; maximum_thr = 2;
                sample_size = 5; est_error = ErrorMetric ::SGD_ERR; break;
            case (EstimationMethod::P3P):
                avg_num_models = 1.38; time_for_model_est = 800;
                sample_size = 3; est_error = ErrorMetric ::RERPOJ; break;
            case (EstimationMethod::P6P):
                avg_num_models = 1; time_for_model_est = 300;
                sample_size = 6; est_error = ErrorMetric ::RERPOJ; break;
            default: CV_Error(cv::Error::StsNotImplemented, "Estimator has not implemented yet!");
        }

        if (estimator_ == EstimationMethod::P3P || estimator_ == EstimationMethod::P6P) {
            neighborsType = NeighborSearchMethod::NEIGH_FLANN_KNN;
            k_nearest_neighbors = 2;
        }
        if (estimator == EstimationMethod::Fundamental || estimator == EstimationMethod::Essential) {
            lo_sample_size = 14;
            lo_thr_multiplier = 10;
        }
        if (estimator == EstimationMethod::Homography)
            maximum_thr = 8.;
        threshold = threshold_;
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
    void setLOIterativeIters (int iters) {lo_iterative_iterations = iters; }
    void setLOSampleSize (int lo_sample_size_) { lo_sample_size = lo_sample_size_; }
    void setThresholdMultiplierLO (double thr_mult) { lo_thr_multiplier = (int)thr_mult; }
    void maskRequired (bool need_mask_) { need_mask = need_mask_; }
    void setPlaneAndParallaxIters (int iters) { plane_and_parallax_max_iters = iters; }
    void setImageSize (cv::Size img_size_) {img_size = img_size_; }
    void setRandomGeneratorState (int state) { random_generator_state = state; }
    void setKmlesac (double k) { k_mlesac = k; }
    void setRansacSolver (MethodSolver s) { r_solver = s; }
    void forceLO (bool force) { force_lo = force; }

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
    int getMaxNumHypothesisToTestBeforeRejection() const {
        return max_hypothesis_test_before_verification;
    }
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
    double getTimeForModelEstimation () const { return time_for_model_est; }
    double getRelaxCoef () const { return relax_coef; }
    const std::vector<int> &getGridCellNumber () const { return grid_cell_number; }
    cv::Size getImageSize () const { return img_size; }
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
    bool isForceLO () const { return force_lo; }
};

    bool estimate (const Params &params, cv::InputArray points1, cv::InputArray points2,
                   Output &output, cv::InputArray K1_, cv::InputArray K2_,
                   cv::InputArray dist_coeff1, cv::InputArray dist_coeff2);
}


#endif //VSAC_PARAMS_HPP
