import numpy as np
from face_detector.configs import configs
import cv2
import face_recognition


desiredLeftEye= (0.35,0.35)
(desiredFaceHeight,desiredFaceWidth) = configs.face_describer_tensor_shape


def align(image,predictor):
		leftEyeCenter=predictor['left_eye']
		rightEyeCenter=predictor['right_eye']
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) 
		desiredRightEyeX = 1.0 - desiredLeftEye[0]

		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - desiredLeftEye[0])
		desiredDist *= desiredFaceWidth
		scale = desiredDist / dist

		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		tX = desiredFaceWidth * 0.5
		tY = desiredFaceHeight * desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		(w, h) = (desiredFaceWidth, desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)

		return output
