import pvsac, numpy as np, cv2, random

folder, name = "samples/data/", "leuven"
try:
    points_file = open(folder+name+"_pts.txt")
except:
    print('File: '+folder+name+"_pts.txt is not found! Try to run from `vsac` directory.")
    exit(1)

num_points = int(points_file.readline())
points = np.array([[float(num) for num in line.split(' ')] for line in points_file])
pts1, pts2 = points[:,0:2].copy(), points[:,2:4].copy()
pts1_val = []
pts2_val = []
pts1_sample = []
pts2_sample = []
for i, (p1, p2) in enumerate(zip(pts1, pts2)):
    if random.randint(0, 2) == 0: # 1/3 probability
        pts1_sample.append(p1)
        pts2_sample.append(p2)
    else:
        pts1_val.append(p1)
        pts2_val.append(p2)
pts1_val = np.array(pts1_val)
pts2_val = np.array(pts2_val)
pts1_sample = np.array(pts1_sample)
pts2_sample = np.array(pts2_sample)
print('size of original points', pts1.shape[0], 'sample set', pts1_sample.shape[0], 'validation set', pts1_val.shape[0])
params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 1.5, 0.99, 3000,
                      pvsac.SamplingMethod.SAMPLING_UNIFORM,
                      pvsac.ScoreMethod.SCORE_METHOD_MSAC)
params.setParallel(True)

model, inliers = pvsac.estimate(params, pts1_sample, pts2_sample, pts1_val=pts1_val, pts2_val=pts2_val)
print(model/model[model.shape[0]-1,model.shape[1]-1], "up-to-scale model, num found inliers", inliers.sum(), "/", pts1_val.shape[0])
