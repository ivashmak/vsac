import pvsac, numpy as np, cv2
from datetime import datetime

def drawInliers (img1, img2, pts1, pts2, inliers_mask):
    img = cv2.hconcat((img1, img2))
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        if inliers_mask[i]:
            cv2.circle(img, (int(p1[0]), int(p1[1])), 10, (0,255,0), -1)
            cv2.circle(img, (int(p2[0])+img1.shape[1], int(p2[1])), 10, (0,255,0), -1)
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0])+img1.shape[1], int(p2[1])), (0,255,0), 2)
    return cv2.resize(img, (int(0.25*img.shape[1]), int(0.25*img.shape[0])))

show_H, show_F, show_E, show_P = True, True, True, True

def runOpenCV (name, pts1, pts2, flag, thr, confidence, iters, K=None):
    start_time = datetime.now()
    if name == 'H':
        model, inliers = cv2.findHomography(pts1, pts2, flag, thr, confidence, iters)
    elif name == 'F':
        model, inliers = cv2.findFundamentalMat(pts1, pts2, flag, thr, confidence, iters)
    elif name == 'E':
        model, inliers = cv2.findEssentialMat(pts1, pts2, K, flag, prob=confidence, threshold=thr) # iters by default 1000
    else:
        print('Not implemented')
        assert False
    runtime = datetime.now() - start_time
    print('OpenCV. Problem', name, ', time (ms) ', runtime.microseconds/1e3, 'number of inliers', inliers.sum())
    return model, inliers

def run(params, pts1, pts2, K1=None, K2=None, dist_coef1=None, dist_coef2=None, img1=None, img2=None):
    start_time = datetime.now()
    model, inliers = pvsac.estimate(params, pts1, pts2, K1, K2, dist_coef1, dist_coef2)
    runtime = datetime.now() - start_time
    print(model, "model, #inliers", inliers.sum(), "/", pts1.shape[0], 'time (ms)', runtime.microseconds/1e3)
    if img1 is not None and img2 is not None:
        cv2.imshow("matches", drawInliers(img1, img2, pts1, pts2, inliers))
        cv2.waitKey(0); cv2.destroyAllWindows()
    return model, inliers

if show_H or show_F or show_F:
    folder, name = "samples/data/", "leuven"
    try:
        points_file = open(folder+name+"_pts.txt")
    except:
        print('File: '+folder+name+"_pts.txt is not found! Try to run from `vsac` directory.")
        exit(1)

    K_file = open(folder+name+"K.txt")
    img1 = cv2.imread(folder+name+"A.jpg")
    img2 = cv2.imread(folder+name+"B.jpg")

    num_points = int(points_file.readline())
    points = np.array([[float(num) for num in line.split(' ')] for line in points_file])
    pts1, pts2 = points[:,0:2].copy(), points[:,2:4].copy()
    K = np.array([[float(num) for num in line.split(' ')] for line in K_file])

    if show_H: # Homography matrix
        params = pvsac.Params(pvsac.EstimationMethod.Homography, 1.5, 0.99, 10000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC)
        # add parallelization, note it is noticeable for higher number of iterations
        # params.setParallel(True)
        H, inliers = run(params, pts1, pts2, img1=img1, img2=img2)
        pts1_corr, pts2_corr = pvsac.getCorrectedPointsHomography(pts1, pts2, H, inliers)
        # runOpenCV('H', pts1, pts2, cv2.RANSAC, 1.5, 0.99, 10000)
    if show_F: # Fundamental matrix
        params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 1., 0.99, 10000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC)
        F, inliers = run(params, pts1, pts2, img1=img1, img2=img2)
        pts1_corr, pts2_corr = pvsac.triangulatePointsLindstrom(F, pts1, pts2, inliers)
        # runOpenCV('F', pts1, pts2, cv2.FM_RANSAC, 1., 0.99, 10000)
if show_E: # Essential matrix
        params = pvsac.Params(pvsac.EstimationMethod.Essential, 1., 0.99, 1000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC)
        E, inliers = run(params, pts1, pts2, K1=K, K2=K, img1=img1, img2=img2)
        pts1_corr, pts2_corr, pts3D, R, t = pvsac.triangulatePointsLindstrom(np.linalg.inv(K).T @ E @np.linalg.inv(K), pts1, pts2, K, K, inliers)
        # runOpenCV('E', pts1, pts2, cv2.RANSAC, 1., 0.99, 1000, K)

if show_P: # P3P
    try:
        data_file = open('samples/data/pnp_scene_from_tless.txt')
    except:
        print('File: samples/data/pnp_scene_from_tless.txt is not found!')
        exit(1)
    scene_id, img_id, obj_id = data_file.readline().split(' ')
    K = np.zeros((3,3)); Rt = np.zeros((3,4))
    for i in range(3):
        K[i] = [float(x) for x in data_file.readline().split(' ')]
    num_poses = int(data_file.readline())
    for i in range(3):
        Rt[i] = [float(x) for x in data_file.readline().split(' ')]
    num_points = int(data_file.readline())
    points_data = np.zeros((num_points, 10))
    for i in range(num_points):
        points_data[i] = [float(x) for x in data_file.readline().split(' ')]
    pts1, pts2 = points_data[:,0:2].copy(), points_data[:,2:5].copy()
    run(pvsac.Params(pvsac.EstimationMethod.P3P, 2.0, 0.99, 3000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC), pts1, pts2, K1=K)
    P_gt = K @ Rt
    print (P_gt, 'Ground Truth Projection matrix')
