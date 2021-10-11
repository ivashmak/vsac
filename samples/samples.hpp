#ifndef VSAC_SAMPLES_HPP
#define VSAC_SAMPLES_HPP

#include <opencv2/core/mat.hpp>

namespace Samples {
    enum DETECTOR { SIFT, SURF, ORB };
    enum ESTIMATION_TASK { AFFINE_MAT, FUNDAMENTAL_MAT, HOMOGRAPHY_MAT, ESSENTIAL_MAT, PROJECTION_MAT_P3P };
    bool getDescriptorsAndKeypoints(Samples::DETECTOR detector_name, const cv::Mat &image, cv::Mat &descriptors, std::vector<cv::KeyPoint> &keypoints, int max_features);
    void getMatches (Samples::DETECTOR detector, const cv::Mat &descriptors1, const cv::Mat &descriptors2,
         const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, cv::Mat &points1, cv::Mat &points2, int kd_trees,
         int flann_search_params, double good_match_ratio, bool sort_by_scores, const std::string &save_fname);

    bool detectCorrespondences (Samples::DETECTOR detector_name, const std::string &img1_name, const std::string &img2_name, cv::Mat &points1, cv::Mat &points2,
        bool sort_by_scores=true, double good_match_ratio=0.7, int kd_trees=5, int flann_search_params=32, const std::string& out_fname="");

    bool detectCorrespondences (Samples::DETECTOR detector_name, const cv::Mat &image1, const cv::Mat &image2, cv::Mat &points1, cv::Mat &points2,
        bool sort_by_scores=true, double good_match_ratio=0.7, int kd_trees=5, int flann_search_params=32, const std::string& out_fname="");

    void exampleDetector ();
    void example (ESTIMATION_TASK task);
    namespace Loader {
        bool readPoints (const std::string &filename, cv::Mat &points1, cv::Mat &points2);
        void readMatrix (cv::Mat &M, int rows, int columns, const std::string &filename);
        void readPnPData (const std::string &fname, cv::Mat &K, cv::Mat &R, cv::Mat &t, cv::Mat &img_pts, cv::Mat &obj_pts);
    };
}

#endif // OPENCV_USAC_SAMPLES_HPP
