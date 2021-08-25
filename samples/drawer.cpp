#include "drawer.hpp"

#include <opencv2/imgproc.hpp>

void Drawer::drawing_resize (cv::Mat &image, int new_img_size) {
    /*
        * w ~ original width;  h ~ original height; S original square;
        * constraint 1: new square is S' = w' * h' == 480000
        * constraint 2: ratio is the same
        *      S'
        * h' = -
        *      w'
        *
        * w'  w             wh'   wS'                  wS'
        * - = -   =>  w' = ---  = --   =>  w' = sqrt (----)
        * h'  h             h     hw'                  h
        *
        *                                              hS'
        *                                  h' = sqrt (----)
        *                                              w
        */
    cv::resize(image, image, cv::Size(sqrt ((double) image.cols * new_img_size / image.rows), sqrt ((double) image.rows * new_img_size / image.cols)));
}

void Drawer::drawHresiduals (const cv::Mat &H, const cv::Mat &points, cv::Mat &img2, const std::vector<int> &inliers, int inliers_size, int circle_size, int line_size, bool rand_colour) {
    const auto * const pts = (float *) points.data;
    cv::Scalar color (0, 0, 255);
    for (int i = 0; i < inliers_size; i++) {
        if (rand_colour) color = cv::Scalar(random()%256, random()%256, random()%256);
        const int inl = inliers[i];
        cv::Vec3d pt1 (pts[4*inl], pts[4*inl+1], 1), pt2(pts[4*inl+2], pts[4*inl+3], 1);
        cv::Mat pt2_est = H * pt1;
        pt2_est /= pt2_est.at<double>(2);
        cv::line(img2, cv::Point2d(pt2[0], pt2[1]), cv::Point2d(pt2_est.at<double>(0), pt2_est.at<double>(1)), color, line_size);
    }
}

/*
    * show correspondences (inliers only), output is concatenated two images horizontally or vertically
    * and lines from point on the first image to point on the second image.
    */
void Drawer::showMatches (const cv::Mat &img1, const cv::Mat &img2, cv::Mat &img12, const std::vector<bool> &mask,
        const cv::Mat &points1_, const cv::Mat &points2_, bool hor, int thickness, bool random_colors, bool show_all_points, int inlier_sz, int offset, cv::Scalar back) {
    cv::Mat points1 = points1_, points2 = points2_;
    points1.convertTo(points1, CV_32F);
    points2.convertTo(points2, CV_32F);
    const auto * const pts1 = (float *) points1.data, * const pts2 = (float *) points2.data;

    int img_height = 0, img_width = 0;
    if (hor) img_width = img1.cols+offset;
    else img_height = img1.rows+offset;
    Drawer::concatenateImage(hor, img1, img2, img12, offset, back);

    if (show_all_points) {
        int outlier_size = ceil(inlier_sz / 2);
        cv::Scalar black = cv::Scalar(0,0,0);
        for (int i = 0; i < points1_.rows; i++) {
            if (mask[i]) continue; // show only outliers
            cv::circle (img12, cv::Point2f(pts1[2*i], pts1[2*i+1]), outlier_size, black, -1);
            cv::circle (img12, cv::Point2f(img_width+pts2[2*i], img_height+pts2[2*i+1]), outlier_size, black, -1);
            // for (int d = 0; d < 3; d++) {
            //     cv::circle (img12, mat2pt(pts.row(i).colRange(0,2)), outlier_size-d, black);
            //     cv::circle (img12, cv::Point_<double> (img_width+pts.at<double>(i, 2), img_height+pts.at<double>(i, 3)), outlier_size-d, black);
            // }
        }
    }

    cv::Scalar color = cv::Scalar(0,255,0);
    for (int i = 0; i < points1.rows; i++) {
        if (!mask[i]) continue;
        if (random_colors)
            color = cv::Scalar(random() % 256, random () % 256, random() % 256);
        cv::Point2f pt1(pts1[2*i], pts1[2*i+1]);
        cv::Point2f pt2 (img_width+pts2[2*i], img_height+pts2[2*i+1]);
        cv::circle (img12, pt1, inlier_sz, color, -1);
        cv::circle (img12, pt2, inlier_sz, color, -1);
        // for (int d = 0; d < 3; d++) {
        //     cv::circle (img12, mat2pt(pts.row(i).colRange(0,2)), inlier_sz-d, color);
        //     cv::circle (img12, cv::Point_<double> (img_width+pts.at<double>(i, 2), img_height+pts.at<double>(i, 3)), inlier_sz-d, color);
        // }
        cv::line(img12, pt1, pt2, color, thickness);
    }
}

void Drawer::drawCorrectedPointsF (cv::Mat &img1, cv::Mat &img2, const cv::Mat &orig_points, const cv::Mat &corr_points, int circle_sz, int line_sz) {
    assert(orig_points.size == corr_points.size);
    const auto * const orig_pts = (float *) orig_points.data, * const corr_pts = (float *) corr_points.data;
    cv::Scalar red (0, 0, 255), green (0, 255, 0), blue (255, 0, 0);
    for (int i = 0; i < orig_points.rows; i++) {
        const auto x1 = orig_pts[4*i], y1 = orig_pts[4*i+1], x2 = orig_pts[4*i+2], y2 = orig_pts[4*i+3];
        const auto X1 = corr_pts[4*i], Y1 = corr_pts[4*i+1], X2 = corr_pts[4*i+2], Y2 = corr_pts[4*i+3];
        cv::circle(img1, cv::Point2d(x1, y1), circle_sz, red);
        cv::circle(img1, cv::Point2d(X1, Y1), circle_sz, green);
        cv::circle(img2, cv::Point2d(x2, y2), circle_sz, red);
        cv::circle(img2, cv::Point2d(X2, Y2), circle_sz, green);

//            cv::line(img1, cv::Point2d(x1, y1), cv::Point2d(X1, Y1), blue, line_sz);
//            cv::line(img2, cv::Point2d(x2, y2), cv::Point2d(X2, Y2), blue, line_sz);
    }
}


void Drawer::drawEpipolarLines (const cv::Mat &F, const cv::Mat &pts1_, const cv::Mat &pts2_, const cv::Mat &img1_, const cv::Mat &img2_,
        cv::Mat &new_image, int max_lines, int line_sz, int circle_sz, const std::vector<int> &inliers, int offset, cv::Scalar back){
    cv::Mat pts1, pts2;
    pts1_.convertTo(pts1, CV_64F);
    pts2_.convertTo(pts2, CV_64F);
    cv::vconcat(pts1.t(), cv::Mat::ones(1, pts1.rows, CV_64F), pts1);
    cv::vconcat(pts2.t(), cv::Mat::ones(1, pts2.rows, CV_64F), pts2);
    cv::Mat image1 = img1_.clone(), image2 = img2_.clone();
    const int pts_size = inliers.empty() ? pts1.rows : (int) inliers.size();
    cv::RNG rng;

    int plot_lines = 0;
    for (int i = 0; i < pts_size; i++) {
        const int pt = inliers.empty() ? i : inliers[i];

        const cv::Scalar col(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        const cv::Mat l2 = F * pts1.col(pt);
        const cv::Mat l1 = F.t() * pts2.col(pt);
        double a1 = l1.at<double>(0), b1 = l1.at<double>(1), c1 = l1.at<double>(2);
        double a2 = l2.at<double>(0), b2 = l2.at<double>(1), c2 = l2.at<double>(2);
        const double mag1 = sqrt(a1 * a1 + b1 * b1), mag2 = (a2 * a2 + b2 * b2);
        a1 /= mag1; b1 /= mag1; c1 /= mag1;
        a2 /= mag2; b2 /= mag2; c2 /= mag2;
        if (plot_lines++ < max_lines) {
            line(image1, cv::Point2d(0, -c1 / b1),
                    cv::Point2d((double) image1.cols, -(a1 * image1.cols + c1) / b1), col, line_sz);
            line(image2, cv::Point2d(0, -c2 / b2),
                    cv::Point2d((double) image2.cols, -(a2 * image2.cols + c2) / b2), col, line_sz);
            circle(image1, cv::Point2d(pts1_.row(pt)), circle_sz, col, -1);
            circle(image2, cv::Point2d(pts2_.row(pt)), circle_sz, col, -1);
        }
        // for (int k = 0; k < 3; k++) {
        //     circle(image1, cv::Point2d(pts1_.row(pt)), circle_sz-k, col);
        //     circle(image2, cv::Point2d(pts2_.row(pt)), circle_sz-k, col);
        // }
//        circle(image1, cv::Point2d(pts1_.row(pt)), circle_sz, col, -1);
//        circle(image2, cv::Point2d(pts2_.row(pt)), circle_sz, col, -1);
    }
    concatenateImage(true, image1, image2, new_image, offset, back);
}

void Drawer::concatenateImage (bool per_row, const cv::Mat &img1, const cv::Mat &img2, cv::Mat &concatenated_img, int offset, cv::Scalar background) {
    auto concatenateRow = [&] (const cv::Mat &i1, const cv::Mat &i2) {
        cv::Mat offset_img(i1.rows, offset, img1.type(), background);
        if (offset == 0) {
            cv::hconcat(i1, i2, concatenated_img);
        } else {
            cv::hconcat(i1, offset_img, concatenated_img);
            cv::hconcat(concatenated_img, i2, concatenated_img);
        }
    };
    auto concatenateCol = [&] (const cv::Mat &i1, const cv::Mat &i2) {
        cv::Mat offset_img(offset, i1.cols, img1.type(), background);
        if (offset == 0) {
            cv::vconcat(i1, i2, concatenated_img);
        } else {
            cv::vconcat(i1, offset_img, concatenated_img);
            cv::vconcat(concatenated_img, i2, concatenated_img);
        }
    };
    if (per_row) {
        if (img1.rows == img2.rows) {
            concatenateRow(img1, img2);
        } else {
            if (img1.rows < img2.rows) {
                cv::Mat img1_ = img1.clone();
                // add zeros matrix which missing to 1 image
                cv::Mat back = cv::Mat(img2.rows - img1.rows, img1.cols, img1.type(), background);
                cv::vconcat(img1_, back, img1_);
                concatenateRow(img1_, img2);
            } else {
                cv::Mat img2_ = img2.clone();
                cv::Mat back = cv::Mat(img1.rows - img2.rows, img2.cols, img2.type(), background);
                cv::vconcat(img2_, back, img2_);
                concatenateRow(img1, img2_);
            }
        }
    } else { // per column
        if (img1.cols == img2.cols) {
            concatenateCol(img1, img2);
        } else {
            if (img1.cols < img2.cols) {
                cv::Mat img1_ = img1.clone();
                cv::Mat back = cv::Mat(img1.rows, img2.cols - img1.cols, img1.type(), background);
                cv::hconcat(img1_, back, img1_);
                concatenateCol(img1_, img2);
            } else {
                cv::Mat img2_ = img2.clone();
                cv::Mat back = cv::Mat(img2.rows, img1.cols - img2.cols, img2.type(), background);
                cv::hconcat(img2_, back, img2_);
                concatenateCol(img1, img2_);
            }
        }
    }
}