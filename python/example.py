import pvsac, numpy as np, cv2

def drawInliers (img1, img2, pts1, pts2, inliers_mask):
    img = cv2.hconcat((img1, img2))
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        if inliers_mask[i]:
            cv2.circle(img, (int(p1[0]), int(p1[1])), 10, (0,255,0), -1)
            cv2.circle(img, (int(p2[0])+img1.shape[1], int(p2[1])), 10, (0,255,0), -1)
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0])+img1.shape[1], int(p2[1])), (0,255,0), 2)
    return cv2.resize(img, (int(0.25*img.shape[1]), int(0.25*img.shape[0])))

show_H, show_F, show_E, show_P = True, True, True, True

def run(params, pts1, pts2, K1=None, K2=None, dist_coef1=None, dist_coef2=None, img1=None, img2=None):
    model, inliers = pvsac.estimate(params, pts1, pts2, K1, K2, dist_coef1, dist_coef2)
    print(model/model[model.shape[0]-1,model.shape[1]-1], "up-to-scale model, num found inliers", inliers.sum(), "/", pts1.shape[0])
    if img1 is not None and img2 is not None:
        cv2.imshow("matches", drawInliers(img1, img2, pts1, pts2, inliers))
        cv2.waitKey(0); cv2.destroyAllWindows()

if show_H or show_F or show_F:
    folder, name = "samples/data/", "leuven"
    try:
        points_file = open(folder+name+"_pts.txt")
    except:
        print('File: '+folder+name+"_pts.txt is not found!")
        exit(1)

    K_file = open(folder+name+"K.txt")
    img1 = cv2.imread(folder+name+"A.jpg")
    img2 = cv2.imread(folder+name+"B.jpg")

    num_points = int(points_file.readline())
    points = np.array([[float(num) for num in line.split(' ')] for line in points_file])
    pts1, pts2 = points[:,0:2].copy(), points[:,2:4].copy()
    K = np.array([[float(num) for num in line.split(' ')] for line in K_file])

    if show_H: # Homography matrix
        run(pvsac.Params(pvsac.EstimationMethod.Homography, 1.5, 0.99, 3000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC),
            pts1, pts2, img1=img1, img2=img2)
    if show_F: # Fundamental matrix
        run(pvsac.Params(pvsac.EstimationMethod.Fundamental, 1.0, 0.99, 3000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC),
            pts1, pts2, img1=img1, img2=img2)
    if show_E: # Essential matrix
        run(pvsac.Params(pvsac.EstimationMethod.Essential, 1.0, 0.99, 3000, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC),
            pts1, pts2, K1=K, K2=K, img1=img1, img2=img2)

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
    print (P_gt / P_gt[2,3], 'Ground Truth Projection matrix')
