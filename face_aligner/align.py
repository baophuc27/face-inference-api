# import numpy as np
# from face_detector.configs import configs
# import cv2
# import face_recognition


# desiredLeftEye= (0.26,0.26)
# (desiredFaceHeight,desiredFaceWidth) = configs.face_describer_tensor_shape


# def align(image,predictor):
# 		leftEyeCenter=predictor['left_eye']
# 		rightEyeCenter=predictor['right_eye']
# 		dY = rightEyeCenter[1] - leftEyeCenter[1]
# 		dX = rightEyeCenter[0] - leftEyeCenter[0]
# 		angle = np.degrees(np.arctan2(dY, dX)) 
# 		desiredRightEyeX = 1.0 - desiredLeftEye[0]

# 		dist = np.sqrt((dX ** 2) + (dY ** 2))
# 		desiredDist = (desiredRightEyeX - desiredLeftEye[0])
# 		desiredDist *= desiredFaceWidth
# 		scale = desiredDist / dist

# 		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
# 			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

# 		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

# 		tX = desiredFaceWidth * 0.5
# 		tY = desiredFaceHeight * desiredLeftEye[1]
# 		M[0, 2] += (tX - eyesCenter[0])
# 		M[1, 2] += (tY - eyesCenter[1])

# 		(w, h) = (desiredFaceWidth, desiredFaceHeight)
# 		output = cv2.warpAffine(image, M, (w, h),
# 			flags=cv2.INTER_CUBIC)

# 		return output
from mtcnn.mtcnn import MTCNN
import cv2
from skimage import transform as trans
import numpy as np

def align(image,landmark):
	detector= MTCNN()
	# result =detector.detect_faces(image)
	# print(landmark)
	landmark = np.asarray(landmark)[0]
	src = np.array([
				[30.2946, 51.6963],
				[65.5318, 51.5014],
				[48.0252, 71.7366],
				[33.5493, 92.3655],
				[62.7299, 92.2041]], dtype=np.float32)
	# landmark= np.asarray(list(result[0]['keypoints'].values()))
	src[:, 0] += 8.0
	dst = landmark.astype(np.float32)
	tform = trans.SimilarityTransform()
	tform.estimate(dst, src)
	M = tform.params[0:2, :]
	warped = cv2.warpAffine(image, M, (112,112), borderValue=0.0)
	cv2.imwrite("face.png",warped)
	return warped