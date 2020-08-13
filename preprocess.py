from mtcnn.mtcnn import MTCNN
import cv2
from skimage import transform as trans
import numpy as np


detector= MTCNN()
image = cv2.imread("rooney_rotate.png")
result =detector.detect_faces(image)
print(result)
src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
landmark= np.asarray(list(result[0]['keypoints'].values()))
src[:, 0] += 8.0
dst = landmark.astype(np.float32)
tform = trans.SimilarityTransform()
tform.estimate(dst, src)
M = tform.params[0:2, :]
warped = cv2.warpAffine(image, M, (112,112), borderValue=0.0)
cv2.imshow("warped",warped)
cv2.waitKey(0)