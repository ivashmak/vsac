#include "precomp.hpp"

namespace cv { namespace vsac {
class TrifocalTensorMinimalSolverImpl : public TrifocalTensorMinimalSolver {
private:
	const Mat &points;
	const float * const pts;
public:
	explicit TrifocalTensorMinimalSolverImpl (const Mat &points_) : points(points_),
		pts((float *) points.data) {}

	static Vec3d getNullVector (const Mat &M) {
		Matx33d u, vt;
		Vec3d d;
		SVDecomp(M, d, u, vt);
		return Vec3d(vt(2,0), vt(2,1), vt(2,2));
	}

	static void getTandEpipoles (const Mat &tensor, std::vector<Matx33d> &T, Vec3d &e_prime, Vec3d &e_prime_prime) {
		T = std::vector<Matx33d> (3);
		int cnt = 0;
		const auto * const ten = (double *) tensor.data;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 3; k++)
					T[i](j,k) = ten[cnt++];
		std::vector<Vec3d> left_null(3), right_null(3);
		for (int i = 0; i < 3; i++) {
			right_null[i] = getNullVector(Mat(T[i]));
			left_null[i] = getNullVector(Mat(T[i].t()));
		}
		/*
		The epipole e′ in the second image is the common intersection of the epipolar lines represented by
		the left null-vectors of the matrices Ti,i = 1,...,3. Similarly the epipole e′′ is the common intersection
		of lines represented by the right null-vectors of the Ti.
		*/
		e_prime = left_null[0].cross(left_null[1]);
		e_prime_prime = right_null[0].cross(right_null[1]);
		e_prime /= norm(e_prime);
		e_prime_prime /= norm(e_prime_prime);
	}
	void getFundamentalMatricesFromTensor (const Mat &tensor, Mat &F21, Mat &F31) override {
		// F21 = [e′]×[T1, T2, T3]e′′ and F31 = [e′′]×[T1^T , T2^T , T3^T ]e′.
		Vec3d e_prime, e_prime_prime;
		std::vector<Matx33d> T;
		getTandEpipoles(tensor, T, e_prime, e_prime_prime);

		const Matx33d e_prime_skew = Math::getSkewSymmetric(e_prime),
				e_prime_prime_skew = Math::getSkewSymmetric(e_prime_prime);

		F21 = Mat_<double>(3,3);
		F31 = Mat_<double>(3,3);
		for (int i = 0; i < 3; i++) {
			Mat(e_prime_skew * T[i] * e_prime_prime).copyTo(F21.col(i));
			Mat(e_prime_prime_skew * T[i].t() * e_prime).copyTo(F31.col(i));
		}
	}

	static void getProjectionsFromTensor (const Mat &tensor, Matx34d &P2, Matx34d &P3) {
		Vec3d a4, b4;
		std::vector<Matx33d> T;
		getTandEpipoles(tensor, T, a4, b4);
		// P′ = [[T1, T2, T3]e′′ | e′] and P′′ = [(e′′e′′T − I)[TT1 , TT2 , TT3 ]e′ | e′′].
		// e' = a4, e'' = b4
		Mat P2_ = Mat_<double>(3,4,P2.val), P3_ = Mat_<double>(3,4,P3.val);
		Mat(a4).copyTo(P2_.col(3));
		Mat(b4).copyTo(P3_.col(3));
		Matx33d bI = (b4 * b4.t() - Matx33d::eye());
		for (int i = 0; i < 3; i++) {
			Mat(T[i] * b4).copyTo(P2_.col(i));
			Mat(bI * T[i].t() * a4).copyTo(P3_.col(i));
		}
	}

	// https://www.robots.ox.ac.uk/~vgg/publications/1997/Torr97a/torr97a.pdf
	Matx14d cross4 (const Vec4d &a, const Vec4d &b, const Vec4d &c) const {
	    const double l1 = determinant(Matx33d(a[1], a[2], a[3],
	    											  b[1], b[2], b[3],
	    											  c[1], c[2], c[3])),
	      			 l2 = -determinant(Matx33d(a[0], a[2], a[3],
	    											   b[0], b[2], b[3],
	    											   c[0], c[2], c[3])),
				     l3 = determinant(Matx33d(a[0], a[1], a[3],
	    											  b[0], b[1], b[3],
	    											  c[0], c[1], c[3])),
				     l4 = -determinant(Matx33d(a[0], a[1], a[2],
	    											   b[0], b[1], b[2],
	    											   c[0], c[1], c[2]));
	    return {l1, l2, l3, l4};
	}

	int estimate(const std::vector<int> &sample, std::vector<Mat> &tensor) const override {
		Matx<double, 3, 5> A;
		auto getLambda = [&] (const int img, Matx33d &B, Vec3d &X5, Vec3d &X6) {
			B = Matx33d (pts[6*sample[0]+2*img  ], pts[6*sample[1]+2*img  ], pts[6*sample[2]+2*img  ],
				  		     pts[6*sample[0]+2*img+1], pts[6*sample[1]+2*img+1], pts[6*sample[2]+2*img+1],
						     1, 1, 1);
			Vec3d x4 (pts[6*sample[3]+2*img], pts[6*sample[3]+2*img+1], 1);
			const auto lambda = B.inv() * x4;
			Mat B_ (3, 3, CV_64F, B.val);
			B_.col(0) *= lambda[0];
			B_.col(1) *= lambda[1];
			B_.col(2) *= lambda[2];
			B = B.inv();
			X5 = B * Vec3d(pts[6*sample[4]+2*img], pts[6*sample[4]+2*img+1], 1);
			X6 = B * Vec3d(pts[6*sample[5]+2*img], pts[6*sample[5]+2*img+1], 1);

		    const double x5 = X5[0], y5 = X5[1], w5 = X5[2], x6 = X6[0], y6 = X6[1], w6 = X6[2];
			A(img, 0) = -x5*y6 + x5*w6;
			A(img, 1) = x6*y5 - y5*w6;
			A(img, 2) = -x6*w5 + y6*w5;
			A(img, 3) = -x5*w6 + y5*w6;
			A(img, 4) = x5*y6 - y6*w5;
		};

		auto getCoeffs = [] (double ai, double bi, double aj, double bj, double ak, double bk) {
			return Vec4d(ai*aj*ak, ai*aj*bk+ai*bj*ak+ai*aj*ak, ai*aj*bk+bi*bj*ak+ai*bj*bk, bi*bj*bk);
		};

		std::vector<Matx33d> Bs(3);
		std::vector<Vec3d> X5s(3), X6s(3);
		getLambda(0, Bs[0], X5s[0], X6s[0]);
		getLambda(1, Bs[1], X5s[1], X6s[1]);
		getLambda(2, Bs[2], X5s[2], X6s[2]);

		std::vector<double> A_vec(A.val, A.val+A.rows*A.cols);
	    vsac::Math::eliminateUpperTriangular(A_vec, A.rows, A.cols);
	    double f1[5], f2[5];

	    f1[4] = 1.;
	    f1[3] = 0.;
	    f1[2] = -A_vec[2*A.cols+4] / A_vec[2*A.cols+2];

	    f2[4] = 1.;
	    f2[3] = -A_vec[2*A.cols+4] / A_vec[2*A.cols+3];
	    f2[2] = 0.;

	    // start from the last row
	    for (int i = A.rows-2; i >= 0; i--) {
	        const int row_i = i*A.cols;
	        double acc1 = 0, acc2 = 0;
	        for (int j = i+1; j < A.cols; j++) {
	            acc1 -= A_vec[row_i + j] * f1[j];
	            acc2 -= A_vec[row_i + j] * f2[j];
	        }
	        f1[i] = acc1 / A_vec[row_i + i];
	        f2[i] = acc2 / A_vec[row_i + i];

	        // due to numerical errors return 0 solutions
	        if (std::isnan(f1[i]) || std::isnan(f2[i]))
	            return 0;
	    }
		const double a1 = f1[0], a2 = f1[1], a3 = f1[2], a4 = f1[3], a5 = f1[4];
		const double b1 = f2[0], b2 = f2[1], b3 = f2[2], b4 = f2[3], b5 = f2[4];

		Vec4d coeffs = getCoeffs(a1,b1,a2,b2,a5,b5)-getCoeffs(a2,b2,a3,b3,a5,b5)
	                -getCoeffs(a2,b2,a4,b4,a5,b5)-getCoeffs(a1,b1,a3,b3,a4,b4)
	                +getCoeffs(a2,b2,a3,b3,a4,b4)+getCoeffs(a3,b3,a4,b4,a5,b5);
	    Vec3d roots;
	    const int nroots = solveCubic (coeffs, roots);
	    if (nroots < 1) return 0;

	    auto getP = [&] (int img, double XbyW, double YbyW, double ZbyW) {
		    const double x5 = X5s[img][0], y5 = X5s[img][1], w5 = X5s[img][2], x6 = X6s[img][0], y6 = X6s[img][1], w6 = X6s[img][2];
		  	Matx44d M (w5, 0, -x5, w5-x5,
		  					0, w5, -y5, w5-y5,
		  					w6*XbyW, 0, -x6*ZbyW, w6-x6,
		  					0, w6*YbyW, -y6*ZbyW, w6-y6);
		  	Mat _D, _U, _Vt;
		  	SVDecomp(M, _D, _U, _Vt, SVD::FULL_UV + SVD::MODIFY_A);
		  	const auto * const _vt = (double *) _Vt.data;
		  	const double a = _vt[3*4], b = _vt[3*4+1], c = _vt[3*4+2], d = _vt[3*4+3];
		  	return  Bs[img].inv() * Matx34d (a, 0, 0, d,
		  										 0, b, 0, d,
		  										 0, 0, c, d);
	    };

	    tensor = std::vector<Mat> (nroots);
	    for (int r = 0; r < nroots; r++) {
	    	const auto root = roots[r];
	    	const double t1 = a1 + root * b1,
				    	 t2 = a2 + root * b2,
				     	 t3 = a3 + root * b3,
				    	 t4 = a4 + root * b4,
				    	 t5 = a5 + root * b5;

		    const double XbyW = (t4-t5)/(t2-t3),
		    			 YbyW = t4/(t1-t3),
		  				 ZbyW = t5/(t1-t2);

		  	Matx34d P1 = getP(0, XbyW, YbyW, ZbyW);
		  	Matx34d P2 = getP(1, XbyW, YbyW, ZbyW);
		  	Matx34d P3 = getP(2, XbyW, YbyW, ZbyW);
		  	Mat P = Mat(P1);

		  	Matx44d H;
		  	vconcat(P1, cross4(P.row(0), P.row(1), P.row(2)), H);
		  	H = H.inv();
		  	P1 = P1 * H;
		  	P2 = P2 * H;
		  	P3 = P3 * H;

		  	tensor[r] = Mat_<double>(27,1);
		  	auto * ten = (double *)tensor[r].data;
		  	for (int i = 0; i < 3; i++)
		  		for (int j = 0; j < 3; j++)
		  			for (int k = 0; k < 3; k++)
		  				(*ten++) = P2(j,i)*P3(k,3) - P2(j,3)*P3(k,i);
	    }
	    return nroots;
	}
    int getMaxNumberOfSolutions () const override { return 3; }
    int getSampleSize() const override { return 6; }
};
Ptr<TrifocalTensorMinimalSolver> TrifocalTensorMinimalSolver::create (const Mat &points_) {
    return makePtr<TrifocalTensorMinimalSolverImpl>(points_);
}

class TrifocalTensorReprErrorImpl : public TrifocalTensorReprError {
private:
    const Mat * points_mat;
    const float * const points;
    double tensor[3][3][3];
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;
    std::vector<float> errors;
public:
	explicit TrifocalTensorReprErrorImpl (const Mat &points_) :
		points_mat(&points_), points ((float*) points_.data), errors(points_.rows)
        , m11(0), m12(0), m13(0), m21(0), m22(0), m23(0), m31(0), m32(0), m33(0) {}
    void setModelParameters (const Mat &model) override {
    	const auto * const m = (double *) model.data;
    	int cnt = 0;
    	for (auto & i : tensor)
    		for (auto & j : i)
    			for (double & k : j)
    				k = m[cnt++];
    }
    float getError (int pidx) const override {
    	pidx *= 6;
    	const auto x1 = points[pidx  ], y1 = points[pidx+1],
    			   x2 = points[pidx+2], y2 = points[pidx+3];
    	const auto z3_ =                   x2 * (x1 * tensor[0][1][2] + y1 * tensor[1][1][2] + tensor[2][1][2]) - y2 * (x1 * tensor[0][0][2] + y1 * tensor[1][0][2] + tensor[2][0][2]);
    	const auto dx3 = points[pidx+4] - (x2 * (x1 * tensor[0][1][0] + y1 * tensor[1][1][0] + tensor[2][1][0]) - y2 * (x1 * tensor[0][0][0] + y1 * tensor[1][0][0] + tensor[2][0][0])) / z3_;
    	const auto dy3 = points[pidx+5] - (x2 * (x1 * tensor[0][1][1] + y1 * tensor[1][1][1] + tensor[2][1][1]) - y2 * (x1 * tensor[0][0][1] + y1 * tensor[1][0][1] + tensor[2][0][1])) / z3_;
    	return dx3 * dx3 + dy3 * dy3;
	}
    const std::vector<float> &getErrors (const Mat &model) override {
    	setModelParameters(model);
    	for (int i = 0; i < points_mat->rows; i++)
    		errors[i] = getError(i);
    	return errors;
    }
};
Ptr<TrifocalTensorReprError> TrifocalTensorReprError::create(const Mat &points) {
    return makePtr<TrifocalTensorReprErrorImpl>(points);
}
}}