#include "samples/samples.hpp"
#include "test/test.hpp"

int main (int /*args*/, char** /*argv*/) {
//    Test::test();

//    Samples::example(Samples::ESTIMATION_TASK::AFFINE_MAT);
    Samples::example(Samples::ESTIMATION_TASK::HOMOGRAPHY_MAT);
//    Samples::example(Samples::ESTIMATION_TASK::FUNDAMENTAL_MAT);
//    Samples::example(Samples::ESTIMATION_TASK::ESSENTIAL_MAT);
//    Samples::example(Samples::ESTIMATION_TASK::PROJECTION_MAT_P3P);
	return 0;
}
