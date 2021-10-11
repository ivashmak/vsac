#include "samples.hpp"

#include <fstream>
#include <iostream>

bool Samples::Loader::readPoints(const std::string &filename, cv::Mat &points1, cv::Mat &points2) {
    std::fstream file(filename);
    if (! file.is_open()) {
        std::cout << "file not open! " << filename << "\n";
        return false;
    }
    int num_points;
    file >> num_points;
    points1 = cv::Mat_<float>(num_points, 2);
    points2 = cv::Mat_<float>(num_points, 2);
    auto * points1_ptr = (float *) points1.data, * points2_ptr = (float *) points2.data;

    for (int i = 0; i < num_points; i++) {
        file >> (*points1_ptr++);
        file >> (*points1_ptr++);
        file >> (*points2_ptr++);
        file >> (*points2_ptr++);
    }
    file.close();
    return true;
}

// read matrix of size rows x columns
void Samples::Loader::readMatrix (cv::Mat &M, int rows, int columns, const std::string &filename) {
    M = cv::Mat_<double>(rows,columns);
    std::fstream file(filename, std::ios_base::in);

    if (! file.is_open()) {
        std::cout << "Wrong direction to matrix file! Reader::readMatrix, " << filename << "\n";
        exit (1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            file >> M.at<double>(i,j);
        }
    }
    file.close();
}

// T-LESS, LMO, YCBV
void Samples::Loader::readPnPData (const std::string &fname, cv::Mat &K, cv::Mat &R, cv::Mat &t,
        cv::Mat &img_pts, cv::Mat &obj_pts) {
    // https://github.com/ducha-aiki/ransac-tutorial-2020-data/blob/master/PnP%20parse%20data.ipynb
    std::ifstream img_file (fname);
    if (!img_file.is_open())
        CV_Error(cv::Error::StsBadArg, "filename not found! "+fname);

    int scene_id, img_id, obj_id, num_poses, pts_size;
    img_file >> scene_id >> img_id >> obj_id;
    K = cv::Mat_<double>(3,3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            img_file >> K.at<double>(i,j);

    img_file >> num_poses;
    assert(num_poses == 1);
    R = cv::Mat_<double>(3,3);
    t = cv::Mat_<double>(3,1);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            if (j == 3) img_file >> t.at<double>(i, 0);
            else img_file >> R.at<double>(i, j);

    img_file >> pts_size;
    img_pts = cv::Mat_<float>(pts_size, 2);
    obj_pts = cv::Mat_<float>(pts_size, 3);
    auto * img_pts_ = (float *) img_pts.data, * obj_pts_ = (float *) obj_pts.data;
    for (int i = 0; i < pts_size; i++) {
        double conf_, conf_obj, conf_frag, pix_id, frag_id;
        img_file >> (*img_pts_++);
        img_file >> (*img_pts_++);
        img_file >> (*obj_pts_++);
        img_file >> (*obj_pts_++);
        img_file >> (*obj_pts_++);
        img_file >> pix_id >> frag_id >> conf_ >> conf_obj >> conf_frag;
    }
    img_file.close();
    // points are sorted by confidence
}
