#include "precomp.hpp"
#define DEBUG_ASPRT false
//#define DEBUG_ASPRT true

namespace cv { namespace vsac {
int Quality::getInliers(const Ptr<Error> &error, const Mat &model, std::vector<int> &inliers, double threshold) {
    const auto &errors = error->getErrors(model);
    int num_inliers = 0;
    for (int point = 0; point < (int)inliers.size(); point++)
        if (errors[point] < threshold)
            inliers[num_inliers++] = point;
    return num_inliers;
}
int Quality::getInliers(const Ptr<Error> &error, const Mat &model, std::vector<bool> &inliers_mask, double threshold) {
    std::fill(inliers_mask.begin(), inliers_mask.end(), false);
    const auto &errors = error->getErrors(model);
    int num_inliers = 0;
    for (int point = 0; point < (int)inliers_mask.size(); point++)
        if (errors[point] < threshold) {
            inliers_mask[point] = true;
            num_inliers++;
        }
    return num_inliers;
}
int Quality::getInliers (const std::vector<float> &errors, std::vector<bool> &inliers, double threshold) {
    std::fill(inliers.begin(), inliers.end(), false);
    int cnt = 0, inls = 0;
    for (const auto e : errors) {
        if (e < threshold) {
            inliers[cnt] = true;
            inls++;
        }
        cnt++;
    }
    return inls;
}
int Quality::getInliers (const std::vector<float> &errors, std::vector<int> &inliers, double threshold) {
    int cnt = 0, inls = 0;
    for (const auto e : errors) {
        if (e < threshold)
            inliers[inls++] = cnt;
        cnt++;
    }
    return inls;
}

class RansacQualityImpl : public RansacQuality {
private:
    const Ptr<Error> error;
    const int points_size;
    const double threshold;
    double best_score;
public:
    RansacQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_)
            : error (error_), points_size(points_size_), threshold(threshold_) {
        best_score = std::numeric_limits<double>::max();
    }

    Score getScore (const Mat &model) const override {
        error->setModelParameters(model);
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++) {
            if (error->getError(point) < threshold)
                inlier_number++;
            if (inlier_number + (points_size - point) < -best_score)
                break;
        }
        // score is negative inlier number! If less then better
        return {inlier_number, -static_cast<double>(inlier_number)};
    }

    Score getScore (const std::vector<float> &errors) const override {
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++)
            if (errors[point] < threshold)
                inlier_number++;
        // score is negative inlier number! If less then better
        return {inlier_number, -static_cast<double>(inlier_number)};
    }

    void setBestScore(double best_score_) override {
        if (best_score > best_score_) best_score = best_score_;
    }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, threshold); }
    double getThreshold () const override { return threshold; }
    int getPointsSize () const override { return points_size; }
    Ptr<Error> getErrorFnc () const override { return error; }
};

Ptr<RansacQuality> RansacQuality::create(int points_size_, double threshold_,
        const Ptr<Error> &error_) {
    return makePtr<RansacQualityImpl>(points_size_, threshold_, error_);
}

class MsacQualityImpl : public MsacQuality {
protected:
    const Ptr<Error> error;
    const int points_size;
    const double threshold, k_msac;
    double best_score, norm_thr, one_over_thr;
public:
    MsacQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_, double k_msac_)
            : error (error_), points_size (points_size_), threshold (threshold_), k_msac(k_msac_) {
        best_score = std::numeric_limits<double>::max();
        norm_thr = threshold*k_msac;
        one_over_thr = 1 / norm_thr;
    }

    inline Score getScore (const Mat &model) const override {
        error->setModelParameters(model);
        double err, sum_errors = 0;
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++) {
            err = error->getError(point);
            if (err < norm_thr) {
                sum_errors -= (1 - err * one_over_thr);
                if (err < threshold)
                    inlier_number++;
            }
            if (sum_errors - points_size + point > best_score)
                break;
        }
        return {inlier_number, sum_errors};
    }

    Score getScore (const std::vector<float> &errors) const override {
        double sum_errors = 0;
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++) {
            const auto err = errors[point];
            if (err < norm_thr) {
                sum_errors -= (1 - err * one_over_thr);
                if (err < threshold)
                    inlier_number++;
            }
        }
        return {inlier_number, sum_errors};
    }

    void setBestScore(double best_score_) override {
        if (best_score > best_score_) best_score = best_score_;
    }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, threshold); }
    double getThreshold () const override { return threshold; }
    int getPointsSize () const override { return points_size; }
    Ptr<Error> getErrorFnc () const override { return error; }
};
Ptr<MsacQuality> MsacQuality::create(int points_size_, double threshold_,
        const Ptr<Error> &error_, double k_msac) {
    return makePtr<MsacQualityImpl>(points_size_, threshold_, error_, k_msac);
}

class MagsacQualityImpl : public MagsacQuality {
private:
    const Ptr<Error> error;
    const Ptr<GammaValues> gamma_generator;
    const int points_size;

    // for example, maximum standard deviation of noise.
    const double maximum_threshold_sqr, tentative_inlier_threshold;
    // The degrees of freedom of the data from which the model is estimated.
    // E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
    const int degrees_of_freedom;
    // A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
    const double k;
    // Calculating k^2 / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double squared_k_per_2;
    // Calculating (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double dof_minus_one_per_two;
    // Calculating (DoF + 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double dof_plus_one_per_two;
    const double C;
    // Calculating 2^(DoF - 1) which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double two_ad_dof_minus_one;
    // Calculating 2^(DoF + 1) which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double two_ad_dof_plus_one;
    // Calculate the gamma value of k
    const double gamma_value_of_k;
    // Calculate the lower incomplete gamma value of k
    const double lower_gamma_value_of_k;
    double previous_best_loss;
    // Convert the maximum threshold to a sigma value
    float maximum_sigma;
    // Calculate the squared maximum sigma
    float maximum_sigma_2;
    // Calculate \sigma_{max}^2 / 2
    float maximum_sigma_2_per_2;
    // Calculate 2 * \sigma_{max}^2
    float maximum_sigma_2_times_2;
    // Calculating 2^(DoF + 1) / \sigma_{max} which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double two_ad_dof_plus_one_per_maximum_sigma;
    double scale_of_stored_incomplete_gammas;
    double max_loss;
    const std::vector<double> &stored_complete_gamma_values, &stored_lower_incomplete_gamma_values;
    int stored_incomplete_gamma_number_min1;
public:

    MagsacQualityImpl (double maximum_thr, int points_size_, const Ptr<Error> &error_,
                       const Ptr<GammaValues> &gamma_generator_,
                       double tentative_inlier_threshold_, int DoF, double sigma_quantile,
                       double upper_incomplete_of_sigma_quantile,
                       double lower_incomplete_of_sigma_quantile, double C_)
            : error (error_), gamma_generator(gamma_generator_), points_size(points_size_),
            maximum_threshold_sqr(maximum_thr*maximum_thr),
            tentative_inlier_threshold(tentative_inlier_threshold_), degrees_of_freedom(DoF),
            k(sigma_quantile), C(C_), gamma_value_of_k (upper_incomplete_of_sigma_quantile),
            lower_gamma_value_of_k (lower_incomplete_of_sigma_quantile),
            stored_complete_gamma_values (gamma_generator->getCompleteGammaValues()),
            stored_lower_incomplete_gamma_values (gamma_generator->getIncompleteGammaValues()) {
        previous_best_loss = std::numeric_limits<double>::max();
        squared_k_per_2 = k * k / 2.0;
        dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
        dof_plus_one_per_two = (degrees_of_freedom + 1.0) / 2.0;
        two_ad_dof_minus_one = std::pow(2.0, dof_minus_one_per_two);
        two_ad_dof_plus_one = std::pow(2.0, dof_plus_one_per_two);
        maximum_sigma = (float)sqrt(maximum_threshold_sqr) / (float) k;
        maximum_sigma_2 = maximum_sigma * maximum_sigma;
        maximum_sigma_2_per_2 = maximum_sigma_2 / 2.f;
        maximum_sigma_2_times_2 = maximum_sigma_2 * 2.f;
        two_ad_dof_plus_one_per_maximum_sigma = two_ad_dof_plus_one / maximum_sigma;
        scale_of_stored_incomplete_gammas = gamma_generator->getScaleOfGammaCompleteValues();
        stored_incomplete_gamma_number_min1 = gamma_generator->getTableSize()-1;

        max_loss = 1e-10;
        // MAGSAC maximum / minimum loss does not have to be in extrumum residuals
        // make 50 iterations to find maximum loss
        const double step = maximum_threshold_sqr / 30;
        double sqr_res = 0;
        while (sqr_res < maximum_threshold_sqr) {
            int x=(int)round(scale_of_stored_incomplete_gammas * sqr_res
                        / maximum_sigma_2_times_2);
            if (x >= stored_incomplete_gamma_number_min1 || x < 0 /*overflow*/)
                x  = stored_incomplete_gamma_number_min1;
            const double loss = two_ad_dof_plus_one_per_maximum_sigma * (maximum_sigma_2_per_2 *
                    stored_lower_incomplete_gamma_values[x] + sqr_res * 0.25 *
                    (stored_complete_gamma_values[x] - gamma_value_of_k));
            if (max_loss < loss)
                max_loss = loss;
            sqr_res += step;
        }
    }

    // https://github.com/danini/magsac
    Score getScore (const Mat &model) const override {
        error->setModelParameters(model);
        double total_loss = 0.0;
        int num_tentative_inliers = 0;
        for (int point_idx = 0; point_idx < points_size; point_idx++) {
            const float squared_residual = error->getError(point_idx);
            if (squared_residual < tentative_inlier_threshold)
                num_tentative_inliers++;
            if (squared_residual < maximum_threshold_sqr) { // consider point as inlier
                // Get the position of the gamma value in the lookup table
                int x=(int)round(scale_of_stored_incomplete_gammas * squared_residual
                        / maximum_sigma_2_times_2);
                // If the sought gamma value is not stored in the lookup, return the closest element
                if (x >= stored_incomplete_gamma_number_min1 || x < 0 /*overflow*/)
                    x  = stored_incomplete_gamma_number_min1;
                // Calculate the loss implied by the current point
                total_loss -= (1 - two_ad_dof_plus_one_per_maximum_sigma * (maximum_sigma_2_per_2 *
                    stored_lower_incomplete_gamma_values[x] + squared_residual * 0.25 *
                    (stored_complete_gamma_values[x] - gamma_value_of_k)) / max_loss);
            }
            if (total_loss - (points_size - point_idx) > previous_best_loss)
                break;
        }
        return {num_tentative_inliers, total_loss};
    }

    Score getScore (const std::vector<float> &errors) const override {
        double total_loss = 0.0;
        int num_tentative_inliers = 0;
        for (int point_idx = 0; point_idx < points_size; point_idx++) {
            const float squared_residual = errors[point_idx];
            if (squared_residual < tentative_inlier_threshold)
                num_tentative_inliers++;
            if (squared_residual < maximum_threshold_sqr) {
                int x=(int)round(scale_of_stored_incomplete_gammas * squared_residual
                                 / maximum_sigma_2_times_2);
                if (x >= stored_incomplete_gamma_number_min1 || x < 0 /*overflow*/)
                    x  = stored_incomplete_gamma_number_min1;
                total_loss -= (1 - two_ad_dof_plus_one_per_maximum_sigma * (maximum_sigma_2_per_2 *
                        stored_lower_incomplete_gamma_values[x] + squared_residual * 0.25 *
                        (stored_complete_gamma_values[x] - gamma_value_of_k)) / max_loss);
            }
            if (total_loss - (points_size - point_idx) > previous_best_loss)
                break;
        }
        return {num_tentative_inliers, total_loss};
    }

    void setBestScore (double best_loss) override {
        if (previous_best_loss > best_loss) previous_best_loss = best_loss;
    }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, tentative_inlier_threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, tentative_inlier_threshold); }
    double getThreshold () const override { return tentative_inlier_threshold; }
    int getPointsSize () const override { return points_size; }
    Ptr<Error> getErrorFnc () const override { return error; }
};
Ptr<MagsacQuality> MagsacQuality::create(double maximum_thr, int points_size_, const Ptr<Error> &error_,
        const Ptr<GammaValues> &gamma_generator,
        double tentative_inlier_threshold_, int DoF, double sigma_quantile,
        double upper_incomplete_of_sigma_quantile,
        double lower_incomplete_of_sigma_quantile, double C_) {
    return makePtr<MagsacQualityImpl>(maximum_thr, points_size_, error_, gamma_generator,
        tentative_inlier_threshold_, DoF, sigma_quantile, upper_incomplete_of_sigma_quantile,
        lower_incomplete_of_sigma_quantile, C_);
}

class LMedsQualityImpl : public LMedsQuality {
private:
    const Ptr<Error> error;
    const int points_size;
    const double threshold;
public:
    LMedsQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_) :
        error (error_), points_size (points_size_), threshold (threshold_) {}

    // Finds median of errors.
    Score getScore (const Mat &model) const override {
        std::vector<float> errors = error->getErrors(model);
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++)
            if (errors[point] < threshold)
                inlier_number++;
        // score is median of errors
        return {inlier_number, Utils::findMedian (errors)};
    }
    Score getScore (const std::vector<float> &errors_) const override {
        std::vector<float> errors = errors_;
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++)
            if (errors[point] < threshold)
                inlier_number++;
        // score is median of errors
        return {inlier_number, Utils::findMedian (errors)};
    }

    void setBestScore (double /*best_score*/) override {}

    int getPointsSize () const override { return points_size; }
    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, threshold); }
    double getThreshold () const override { return threshold; }
    Ptr<Error> getErrorFnc () const override { return error; }
};
Ptr<LMedsQuality> LMedsQuality::create(int points_size_, double threshold_, const Ptr<Error> &error_) {
    return makePtr<LMedsQualityImpl>(points_size_, threshold_, error_);
}

class ModelVerifierImpl : public ModelVerifier {
private:
    std::vector<float> errors;
public:
    inline bool isModelGood(const Mat &/*model*/) override { return true; }
    inline bool getScore(Score &/*score*/) const override { return false; }
    void update (int /*highest_inlier_number*/) override {}
    void updateSPRT (double , double , double , double , double , const Score &) override {}
};
Ptr<ModelVerifier> ModelVerifier::create() {
    return makePtr<ModelVerifierImpl>();
}

class AdaptiveSPRTImpl : public AdaptiveSPRT {
private:
    RNG rng;
    const Ptr<Error> err;
    const Ptr<Quality> quality;
    const int points_size;
    int max_iters_to_adapt, highest_inlier_number, current_sprt_idx;
    // time t_M needed to instantiate a model hypothesis given a sample
    // Let m_S be the number of models that are verified per sample
    const double inlier_threshold, norm_thr, one_over_thr;

    // alpha is false negative rate, alpha = 1 / A
    double t_M, lowest_sum_errors, current_epsilon, current_delta, current_A,
        delta_to_epsilon, complement_delta_to_complement_epsilon,
        time_ver_corr_sprt = 0, time_ver_corr = 0, m_S,
        one_over_complement_alpha, C, avg_num_checked_pts;

    std::vector<SPRT_history> sprt_histories;
    std::vector<int> points_random_pool;
    std::vector<float> errors;

    Score score;
    const ::vsac::ScoreMethod score_type;
    bool do_sprt, last_model_is_good, adapt, IS_ADAPTIVE;
public:
    AdaptiveSPRTImpl (int state, const Ptr<Error> &err_, const Ptr<Quality> &quality_, int points_size_,
              double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
              double time_sample, double avg_num_models, ::vsac::ScoreMethod score_type_, int max_iters_to_adapt_,
              double k_mlesac_, bool is_adaptive) : rng(state), err(err_),
              quality(quality_), points_size(points_size_), max_iters_to_adapt (max_iters_to_adapt_),
                inlier_threshold (quality->getThreshold()), norm_thr(inlier_threshold_*k_mlesac_),
                one_over_thr (1/norm_thr), t_M (time_sample), score_type (score_type_), m_S (avg_num_models) {

        // Generate array of random points for randomized evaluation
        points_random_pool = std::vector<int> (points_size_);
        // fill values from 0 to points_size-1
        for (int i = 0; i < points_size; i++)
            points_random_pool[i] = i;
        randShuffle(points_random_pool, 1, &rng);

        // reserve (approximately) some space for sprt vector.
        sprt_histories.reserve(20);

        highest_inlier_number = 0;
        lowest_sum_errors = std::numeric_limits<double>::max();
        last_model_is_good = false;
        if (score_type_ != ::vsac::ScoreMethod::SCORE_METHOD_MSAC)
            errors = std::vector<float>(points_size_);
        IS_ADAPTIVE = is_adaptive;
        if (IS_ADAPTIVE) {
            adapt = true; do_sprt = false;
            // all these variables will be initialized later
            current_sprt_idx = 0; current_epsilon = prob_pt_of_good_model;
            current_delta = prob_pt_of_bad_model; current_A = -1;
            delta_to_epsilon = -1; C = -1; avg_num_checked_pts = points_size_;
            one_over_complement_alpha = -1; complement_delta_to_complement_epsilon = -1;
        } else {
            adapt = false; do_sprt = true; max_iters_to_adapt = 0;
            createTest(prob_pt_of_good_model, prob_pt_of_bad_model);
        }
    }

    inline bool isModelGood (const Mat &model) override {
        // update error object with current model
        last_model_is_good = true;
        double sum_errors = 0;
        int tested_inliers = 0;
        if (! do_sprt || adapt) { // if adapt or not sprt then compute model score directly
            score = quality->getScore(model);
            tested_inliers = score.inlier_number;
            sum_errors = score.score;
        } else { // do sprt and not adapt
            err->setModelParameters(model);
            double lambda = 1;
            int random_pool_idx = rng.uniform(0, points_size), tested_point;
            if (score_type == ::vsac::ScoreMethod::SCORE_METHOD_MSAC) {
                for (tested_point = 0; tested_point < points_size; tested_point++) {
                    if (random_pool_idx >= points_size)
                        random_pool_idx = 0;
                    const float error = err->getError (points_random_pool[random_pool_idx++]);
                    if (error < inlier_threshold) {
                        tested_inliers++;
                        lambda *= delta_to_epsilon;
                    } else {
                        lambda *= complement_delta_to_complement_epsilon;
                        // since delta is always higher than epsilon, then lambda can increase only
                        // when point is not consistent with model
                        if (lambda > current_A)
                            break;
                    }
                    if (error < norm_thr)
                        sum_errors -= (1 - error * one_over_thr);
                    if (sum_errors - points_size + tested_point > lowest_sum_errors)
                        break;
                }
            } else { // save errors into array here
                for (tested_point = 0; tested_point < points_size; tested_point++) {
                    if (random_pool_idx >= points_size)
                        random_pool_idx = 0;
                    const int pt = points_random_pool[random_pool_idx++];
                    const float error = err->getError (pt);
                    if (error < inlier_threshold) {
                        tested_inliers++;
                        lambda *= delta_to_epsilon;
                    } else {
                        lambda *= complement_delta_to_complement_epsilon;
                        // since delta is always higher than epsilon, then lambda can increase only
                        // when point is not consistent with model
                        if (lambda > current_A)
                            break;
                    }
                    if (tested_inliers + points_size - tested_point < highest_inlier_number)
                        break;
                    errors[pt] = error;
                }
            }
            last_model_is_good = tested_point == points_size;
        }

        // increase number of samples processed by current test
        sprt_histories[current_sprt_idx].tested_samples++;
        if (last_model_is_good) {
            if (do_sprt) {
                score.inlier_number = tested_inliers;
                if (score_type == ::vsac::ScoreMethod::SCORE_METHOD_MSAC)
                    score.score = sum_errors;
                else if (score_type == ::vsac::ScoreMethod::SCORE_METHOD_RANSAC)
                    score.score = -static_cast<double>(tested_inliers);
                else score = quality->getScore(errors);
            }
            const double new_epsilon = static_cast<double>(tested_inliers) / points_size;
            if (new_epsilon > current_epsilon) {
                highest_inlier_number = tested_inliers; // update max inlier number
                createTest(new_epsilon, current_delta);
            }
            if (lowest_sum_errors > sum_errors)
                lowest_sum_errors = sum_errors;
        }
#if DEBUG_ASPRT
        if (! last_model_is_good) {
            const auto m_score = quality->getScore(model);
            if (m_score.inlier_number > highest_inlier_number) {
                std::cout << "MODEL WITH HIGHEST NUM OF INLIERS REJECTED (" << m_score.score << ", " << m_score.inlier_number << ") BEST " << highest_inlier_number << " FN ratio " << 1 / sprt_histories[current_sprt_idx].A << '\n';
            }
        }
#endif
        return last_model_is_good;
    }

    inline bool getScore (Score &score_) const override {
        score_ = score;
        return true;
    }
    // update SPRT parameters = called only once inside VSAC
    void updateSPRT (double time_model_est, double time_corr_ver, double new_avg_models, double new_delta, double new_epsilon, const Score &best_score) override {
        if (adapt) {
            adapt = false;
            t_M = time_model_est / time_corr_ver;
            m_S = new_avg_models;
            time_ver_corr = time_corr_ver;
            time_ver_corr_sprt = 1.05 * time_corr_ver;
            createTest(new_epsilon, new_delta);
            lowest_sum_errors = best_score.score;
            highest_inlier_number = best_score.inlier_number;
        }
    }

    const std::vector<SPRT_history> &getSPRTvector () const override { return sprt_histories; }
    void update (int highest_inlier_number_) override {
        if (!adapt) {
            const double new_epsilon = static_cast<double>(highest_inlier_number_) / points_size;
            if (new_epsilon > current_epsilon) {
                highest_inlier_number = highest_inlier_number_;
                if (sprt_histories[current_sprt_idx].tested_samples == 0)
                    sprt_histories[current_sprt_idx].tested_samples = 1;
                // save sprt test and create new one
                createTest(new_epsilon, current_delta);
            }
        }
    }
private:

    // Saves sprt test to sprt history and update current epsilon, delta and threshold.
    void createTest (double epsilon, double delta) {
        // if epsilon is closed to 1 then set them to 0.99 to avoid numerical problems
        if (epsilon > 0.999999) epsilon = 0.999;
        // delta can't be higher than epsilon, because ratio delta / epsilon will be greater than 1
        if (epsilon < delta) delta = epsilon-0.001;
        // avoid delta going too high as it is very unlikely
        // e.g., 30% of points are consistent with bad model is not very real
        if (delta   > 0.3) delta = 0.3;

        SPRT_history new_sprt_history;
        new_sprt_history.epsilon = epsilon;
        new_sprt_history.delta = delta;
        new_sprt_history.A = estimateThresholdA (epsilon, delta);

        one_over_complement_alpha = 1 / (1 - 1 / new_sprt_history.A);

        sprt_histories.emplace_back(new_sprt_history);

        current_A = new_sprt_history.A;
        current_delta = delta;
        current_epsilon = epsilon;

        delta_to_epsilon = delta / epsilon;
        complement_delta_to_complement_epsilon = (1 - delta) / (1 - epsilon);
        current_sprt_idx = static_cast<int>(sprt_histories.size()) - 1;

        if (IS_ADAPTIVE) {
            avg_num_checked_pts = log(new_sprt_history.A) / C;
            if (1 / new_sprt_history.A > 0.5) do_sprt = false; // if false rejection rate is higher than 50%, do not do SPRT
            else do_sprt = time_ver_corr_sprt * avg_num_checked_pts * one_over_complement_alpha < time_ver_corr * points_size;
        }
#if DEBUG_ASPRT
    std::cout << "AVG CHECK PTS " << avg_num_checked_pts << " / " << points_size << " DO SPRT " << do_sprt << " 1/alpha^ " << one_over_complement_alpha <<
        " T_ver_pt " << time_ver_corr << " DIFF(T) " << (time_ver_corr * points_size / (time_ver_corr_sprt * avg_num_checked_pts * one_over_complement_alpha)) <<
            " DEL " << delta << " EPS " << epsilon << "\n";
#endif
    }
    double estimateThresholdA (double epsilon, double delta) {
        C = (1 - delta) * log ((1 - delta) / (1 - epsilon)) + delta * log (delta / epsilon);
        // K = K1/K2 + 1 = (t_M / P_g) / (m_S / (C * P_g)) + 1 = (t_M * C)/m_S + 1
        const double K = t_M * C / m_S + 1;
        double An, An_1 = K;
        // compute A using a recursive relation
        // A* = lim(n->inf)(An), the series typically converges within 4 iterations
        for (int i = 0; i < 10; i++) {
            An = K + log(An_1);
            if (fabs(An - An_1) < FLT_EPSILON)
                break;
            An_1 = An;
        }
        return An;
    }
};
Ptr<AdaptiveSPRT> AdaptiveSPRT::create (int state, const Ptr<Error> &err_, const Ptr<Quality> &quality, int points_size_,
            double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
            double time_sample, double avg_num_models, ::vsac::ScoreMethod score_type_, int max_iters_to_adapt, double k_mlesac, bool is_adaptive) {
    return makePtr<AdaptiveSPRTImpl>(state, err_, quality, points_size_, inlier_threshold_,
         prob_pt_of_good_model, prob_pt_of_bad_model, time_sample, avg_num_models, score_type_, max_iters_to_adapt, k_mlesac, is_adaptive);
}
}}