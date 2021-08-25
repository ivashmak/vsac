#ifndef VSAC_SAMPLES_DRAWING_HPP
#define VSAC_SAMPLES_DRAWING_HPP

#include <opencv2/core.hpp>

namespace Drawer {
    void drawing_resize (cv::Mat &image, int new_img_size=480000);
    void drawHresiduals (const cv::Mat &H, const cv::Mat &points, cv::Mat &img2, const std::vector<int> &inliers, int inliers_size, int circle_size, int line_size, bool rand_colour);
    /*
     * show correspondences (inliers only), output is concatenated two images horizontally or vertically
     * and lines from point on the first image to point on the second image.
     */
    void showMatches (const cv::Mat &img1, const cv::Mat &img2, cv::Mat &img12, const std::vector<bool> &mask,
            const cv::Mat &pts1, const cv::Mat &pts2, bool hor, int thickness, bool random_colors, bool show_all_points=true, int inlier_sz=15, int offset=0, cv::Scalar back=cv::Scalar(0,0,0));
    void drawCorrectedPointsF (cv::Mat &img1, cv::Mat &img2, const cv::Mat &orig_points, const cv::Mat &corr_points, int circle_sz, int line_sz);    
    void drawEpipolarLines (const cv::Mat &F, const cv::Mat &pts1_, const cv::Mat &pts2_, const cv::Mat &img1_, const cv::Mat &img2_, cv::Mat &new_image, int max_lines, int line_sz, int circle_sz, const std::vector<int> &inliers, int offset=0, cv::Scalar back=cv::Scalar(0,0,0));
    void concatenateImage (bool per_row, const cv::Mat &img1, const cv::Mat &img2, cv::Mat &concatenated_img, int offset=0, cv::Scalar background=cv::Scalar(0,0,0));
};

#endif // OPENCV_USAC_SAMPLES_DRAWING_HPP
