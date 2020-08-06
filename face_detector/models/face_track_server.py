import cv2
import face_recognition
import numpy as np

class FaceTrackServer(object):

    face_landmarks = {}
    cam_h = None
    cam_w = None

    def __init__(self, down_scale_factor=0.3):
        assert 0 <= down_scale_factor <= 1
        self.down_scale_factor = down_scale_factor

    def reset(self):
        self.face_relative_locations = []
        self.face_locations = []
        self.faces = []

    def process_lm(self, frame):
        self.reset()
        self.cam_h, self.cam_w,_= frame.shape

        small_frame = cv2.resize(
            frame, (0, 0), fx=self.down_scale_factor, fy=self.down_scale_factor
        )

        rgb_small_frame = small_frame[:, :, ::-1]
        _face_landmarks = face_recognition.face_landmarks(rgb_small_frame)
        if len(_face_landmarks) > 0:
            left_eye = _face_landmarks[0]["left_eyebrow"]
            right_eye = _face_landmarks[0]["right_eyebrow"]
            leftEyeCenter = np.average(left_eye, axis=0).astype(int)
            rightEyeCenter = np.average(right_eye, axis=0).astype(int)
            (xL, yL) = (leftEyeCenter/self.down_scale_factor).astype(int)
            (xR, yR) = (rightEyeCenter/self.down_scale_factor).astype(int)
            
            xL = xL / self.cam_w
            xR = xR /self.cam_w
            yL = yL/ self.cam_h
            yR = yR / self.cam_h

            self.face_landmarks["left_eye"]=(xL,yL)
            self.face_landmarks["right_eye"]= (xR,yR)

        return self.face_landmarks


    def get_face_landmarks(self):

        return self.face_landmarks
