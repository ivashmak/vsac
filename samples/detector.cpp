#include "samples.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>

bool Samples::getDescriptorsAndKeypoints(DETECTOR detector_name, const cv::Mat &image, cv::Mat &descriptors, std::vector<cv::KeyPoint> &keypoints, int max_features) {
    if (detector_name == DETECTOR::SIFT) {
        auto detector = cv::SIFT::create(max_features);
        detector->detect(image, keypoints);
        detector->compute(image, keypoints, descriptors);
    } else if (detector_name == DETECTOR::ORB) {
        auto detector = cv::ORB::create (max_features);
        detector->detect(image, keypoints);
        detector->compute(image, keypoints, descriptors);
    } else {
        std::cout << "Current detector has not yet implmented! Error message from detectCorrespondences!\n";
        return false;
    }
    if (keypoints.empty()) {
        std::cout << "NOT ENOUGH FEATURES! Error message from detector::detectCorrespondences!\n";
        return false;
    }
    return true;
}
void Samples::getMatches (DETECTOR detector_name, const cv::Mat &descriptors1, const cv::Mat &descriptors2,
        const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, cv::Mat &points1, cv::Mat &points2, int kd_trees,
        int flann_search_params, double good_match_ratio, bool sort_by_scores, const std::string &save_fname) {

    if (detector_name == DETECTOR::SIFT || detector_name == DETECTOR::SURF) {
        cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(kd_trees), new cv::flann::SearchParams(flann_search_params));

        // get k=2 best match that we can apply ratio test explained by D.Lowe
        // https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf, page=20
        std::vector<std::vector<cv::DMatch>> matches_vector;
        matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

        cv::Mat pts = cv::Mat_<float>(matches_vector.size(), 4);
        auto * pts_ = (float *) pts.data;

        std::vector<double> scores;
        scores.reserve(matches_vector.size());

        // std::cout << "Find good matches\n";
        for (const auto &m : matches_vector) {
            // compare best and second match using Lowe ratio test
            if (m[0].distance / m[1].distance < good_match_ratio) {
                const cv::Point2d &p1 = keypoints1[m[0].queryIdx].pt;
                const cv::Point2d &p2 = keypoints2[m[0].trainIdx].pt;
                (*pts_++) = p1.x;
                (*pts_++) = p1.y;
                (*pts_++) = p2.x;
                (*pts_++) = p2.y;
                // store Lowe ratio as score
                scores.emplace_back(m[0].distance / m[1].distance);
            }
        }
        pts.rowRange(0, (int)scores.size()).copyTo(pts);

        if (sort_by_scores) {
            std::vector<int> sorted_mask (scores.size());
            std::iota(sorted_mask.begin(), sorted_mask.end(), 0);

            // sort sorted_mask using scores by increasing score
            std::sort (sorted_mask.begin(), sorted_mask.end(), [&] (int a, int b) {
                return scores[a] < scores[b];
            });

            // store points to cvMat
            points1 = cv::Mat_<float> (pts.rows, 2);
            points2 = cv::Mat_<float> (pts.rows, 2);
            auto * points1_ptr = (float *) points1.data, * points2_ptr = (float *) points2.data;
            const auto * const pts_data = (float *) pts.data;
            for (const auto &sorted_idx : sorted_mask) {
                *(points1_ptr++) = pts_data[4 * sorted_idx    ];
                *(points1_ptr++) = pts_data[4 * sorted_idx + 1];
                *(points2_ptr++) = pts_data[4 * sorted_idx + 2];
                *(points2_ptr++) = pts_data[4 * sorted_idx + 3];
            }
        } else {
            pts.colRange(0,2).copyTo(points1);
            pts.colRange(2,4).copyTo(points2);
        }

        if (!save_fname.empty()) {
            std::ofstream fname (save_fname);
            const auto * const _pts1 = (float *) points1.data, * const _pts2 = (float *) points2.data;
            for (int i = 0; i < points1.rows; i++) {
                fname << _pts1[2*i] << " "<< _pts1[2*i+1] << " "<< _pts2[2*i] << " "<< _pts2[2*i+1] << "\n";
            }
            fname.close();
        }
    } else {
        std::cout << "Matching is not yet implemented for the current keypoint detector!\n";
        exit(1);
    }
}

bool Samples::detectCorrespondences (DETECTOR detector_name, const std::string &img1_name, const std::string &img2_name, cv::Mat &points1, cv::Mat &points2,
        bool sort_by_scores, double good_match_ratio, int kd_trees, int flann_search_params,
        const std::string& out_sorted_pts_file) {

    const cv::Mat image1 = cv::imread(img1_name), image2 = cv::imread(img2_name);

    if (image1.empty() || image2.empty()) {
        std::cout << "Images empty! Pathes: " << img1_name << ", " << img2_name << "\n"; 
        return false;
    }

    return detectCorrespondences(detector_name, image1, image2, points1, points2, sort_by_scores, good_match_ratio, kd_trees, flann_search_params, out_sorted_pts_file);
}

bool Samples::detectCorrespondences (DETECTOR detector_name, const cv::Mat &image1, const cv::Mat &image2, cv::Mat &points1, cv::Mat &points2,
        bool sort_by_scores, double good_match_ratio, int kd_trees, int flann_search_params, const std::string& out_sorted_pts_file) {

    cv::Mat descriptors1, descriptors2;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    getDescriptorsAndKeypoints(detector_name, image1, descriptors1, keypoints1, 0);
    getDescriptorsAndKeypoints(detector_name, image2, descriptors2, keypoints2, 0);
    getMatches(detector_name, descriptors1, descriptors2, keypoints1, keypoints2, points1, points2, kd_trees,
        flann_search_params, good_match_ratio, sort_by_scores, out_sorted_pts_file);
    return true;
}
