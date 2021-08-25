#include "samples.hpp"

#include "../include/vsac.hpp"
#include <iostream>

#include "drawer.hpp"
#include <opencv2/highgui.hpp>

void Samples::exampleDetector () {
    const std::string &name = "leuven";
    // run example from build directory
    const std::string &img1_name = "../samples/data/"+name+"A.jpg";
    const std::string &img2_name = "../samples/data/"+name+"B.jpg";

    // try to read points from the file
    // file format is:
    // N
    // x11 y11 x12 y12
    // x21 y21 x22 y22
    // ...
    // xN1 yN1 xN2 yN2

    cv::Mat points1, points2;
    // if file not exists then find features and save them to file
    if (!detectCorrespondences(DETECTOR::SIFT, img1_name, img2_name, points1, points2,
             true /*sort pts by Lowe ratio test*/,
             0.75 /* good point coefficient*/,
             5 /*kd-trees for opencv flann matched*/, 32 /*flann opencv search parameters*/,
             name+"_pts.txt")) {
        std::cerr << "Failed to find correspondences!\n";
        return;
    }

    int points_size = points1.rows;
    std::cout << "Number of found matches: " << points_size << "\n";
}

void Samples::example (ESTIMATION_TASK task) {
    std::string folder = "../samples/data/", fname;
    cv::Mat points1, points2, K1, K2, R, t, model, distortion_coeff1, distortion_coeff2;
    double threshold, confidence = 0.99;
    const bool sorted = true;
    const int max_iters = 2000;
    vsac::EstimationMethod est_task;
    if (task == ESTIMATION_TASK::PROJECTION_MAT_P3P) {
        est_task = vsac::EstimationMethod ::P3P;
        threshold = 2.;
        Loader::readPnPData(folder + "pnp_scene_from_tless.txt", K1, R, t, points1, points2);
    } else {
        fname = "leuven";
        // USE separate points as in PnP
        // read points sorted by SIFT ratio
        if (!Loader::readPoints(folder + fname + "_pts.txt", points1, points2)) {
            std::cerr << "Cannot read points for the filename " << folder + fname + "_pts.txt" << "\n";
            exit(EXIT_FAILURE);
        }
        if (task == ESTIMATION_TASK::HOMOGRAPHY_MAT || task == ESTIMATION_TASK::AFFINE_MAT) {
            threshold = 2.;
            est_task = task == ESTIMATION_TASK ::HOMOGRAPHY_MAT ? vsac::EstimationMethod ::Homography : vsac::EstimationMethod ::Affine;
        } else if (task == ESTIMATION_TASK::FUNDAMENTAL_MAT || task == ESTIMATION_TASK::ESSENTIAL_MAT) {
            threshold = 0.8;
            Loader::readMatrix(K1, 3, 3, folder + fname +"K.txt");
            K1.copyTo(K2); // same intrinsic matrix for this pair of image
            est_task = task == ESTIMATION_TASK ::FUNDAMENTAL_MAT ? vsac::EstimationMethod ::Fundamental : vsac::EstimationMethod ::Essential;
        } else {
            std::cerr << "The estimation task is currently not implemented!\n";
            exit(EXIT_FAILURE);
        }
    }
    // advanced settings and parameters
    vsac::Params params (threshold, est_task, sorted ? vsac::SamplingMethod::SAMPLING_PROSAC : vsac::SamplingMethod::SAMPLING_UNIFORM,
         confidence, max_iters, vsac::ScoreMethod::SCORE_METHOD_MSAC); // worth trying also SCORE_METHOD_MAGSAC, todo: change MSAC to MLESAC
    
    /////////////////////////////////////////////// OPTIONAL PARAMETERS //////////////////////////////
    // params.setRandomGeneratorState(random() % INT_MAX); // set state of the random generator
    // LO worth trying LOCAL_OPTIM_GC or LOCAL_OPTIM_SIGMA
    // params.setLocalOptimization(vsac::LocalOptimMethod ::IRLS);
    // params.setVerifier(vsac::VerificationMethod::NullVerifier); // if you get bad results try to switch to NullVerifier
    // params.setPolisher(vsac::PolishingMethod::MAGSAC); // try also MAGSAC which is more accurate, but slower

    // parameters for LO such number of iterations and non-minimal sample size
    // higher parameters -> slower, maybe better results
    // params.setLOSampleSize(5*params.getSampleSize());
    // params.setLOIterations(12);

    // try to parallelize. The speed up is more noticeable when points are not sorted
    // params.setParallel(true);

    // bad name of function, but it does change sigma (noise level) for MLESAC score.
    // Threshold decides for inlier / outlier, while sigma change the score of model
    // see src/quality.cpp MsacQuality class
    // params.setKmlesac(2.5); // try to increase MLESAC scale

    // if (task == ESTIMATION_TASK::HOMOGRAPHY_MAT)
    //     params.setError(vsac::ErrorMetric::FORW_REPR_ERR); // you can also try SYMM_REPR_ERR
         // error (distance) metric. Default is forard reprojection error ||x' - Hx||^2
        // you also try SYMM_REPR_ERR -- symmetric which is ||x' - Hx||^2 + ||x - H^-1x'||^2
    // else if (task == ESTIMATION_TASK::ESSENTIAL_MAT || task == ESTIMATION_TASK::FUNDAMENTAL_MAT)
    //     params.setError(vsac::ErrorMetric::SAMPSON_ERR); // you can also try SGD (symmetric geometric distance)

    // if you do not need inlier mask in the end
    // params.maskRequired(false);

    // if (params.isFundamental() || params.isEssential())
    //    params.setEnforceRank(false); // a model with better fit of inliers can be obtained by switching rank constraint off

    // Uncomment to set a model non-randomness test. In the output you can check whether the final model is random
    // params.setNonRandomnessTest(true);

    // EXPERIMENTAL, if uniform sampling is used, try to use quasi-uniform sampling.
    // When model is updated, it makes floor(CURRENT_INLIER_SIZE / MINIMAL_SAMPLE_SIZE) samples from inliers of the best model so far.
    // after the standard sampling is back. The results may be better.
    // if (params.getSampler() == vsac::SamplingMethod::SAMPLING_UNIFORM)
    //    params.setQuasiSampling(true);

    // you can check other relevenat function if include/opencv2/usac/usac.hpp on line 825
    // you can check flags for LO, score or sampler in include/opencv2/usac/usac.hpp on line 11
    // you don't have to set such settings every time, most of them are used by default
    // I just showed them that you can play with them
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    vsac::Output output;
    std::vector<bool> inliers_mask;
    std::vector<int> inliers_indicies;
    std::vector<float> point_residuals;
//    try {
        const auto begin_time = std::chrono::steady_clock::now();
        const bool success = vsac::estimate(params, points1, points2, output,
                                            K1, K2, distortion_coeff1, distortion_coeff2);
        const auto time_ms = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_time).count()/1e3;
        if (success) {
            model = output.getModel().clone();
            if (params.isMaskRequired()) {
                point_residuals = output.getPointResiduals();
                inliers_mask = output.getInliersMask();
                inliers_indicies = output.getInliers();
            }
            // check for function of ransac output in include/opencv2/usac/usac.hpp on line 790
            std::cout << "Number of inliers " << output.getNumberOfInliers() << " / " << points1.rows << "; Number of iterations " <<
                      output.getNumberOfIterations() << " / " << max_iters << "; Time (ms) " << time_ms << "\n";
            std::cout << "Found up-to-scale model:\n" << model / model.at<double>(model.rows-1,model.cols-1) << "\n";
            if (params.isNonRandomnessTest()) {
                std::cout << "The final model is " << (output.isModelRandom() ? "random\n" : "non-random\n");
            }
        } else {
            std::cerr << "VSAC failed!\n";
            exit(EXIT_FAILURE);
        }
//    } catch (const std::exception &e) {
//        std::cerr << "VSAC crashed! " << e.what() << "\n";
//        exit(EXIT_FAILURE);
//    }

    if (task == ESTIMATION_TASK::HOMOGRAPHY_MAT || task == ESTIMATION_TASK::FUNDAMENTAL_MAT || task == ESTIMATION_TASK::ESSENTIAL_MAT) {
        cv::Mat corrected_points;
        if (task == ESTIMATION_TASK::HOMOGRAPHY_MAT) {

        } else {
            cv::Mat F = task == ESTIMATION_TASK::ESSENTIAL_MAT ? K2.inv().t() * model = K1.inv() : model;
        }
    }

    if (task != ESTIMATION_TASK::PROJECTION_MAT_P3P) {
        const cv::Mat img1 = cv::imread(folder + fname + "A.jpg"), img2 = cv::imread(folder + fname + "B.jpg");
        assert(!img1.empty() && !img2.empty());

        cv::Mat img12;
        Drawer::showMatches(img1, img2, img12, inliers_mask, points1, points2, true /*horizontal*/, 2 /*line size*/, true /*rand colours*/, true /*show all pts*/, 8/*circle size*/, 80/*offset*/, cv::Scalar(255,255,255) /*background color*/);
        Drawer::drawing_resize(img12, 1200*900);
        cv::imshow("inliers", img12);
        cv::waitKey(0);
    }
}
