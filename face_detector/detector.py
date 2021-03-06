import cv2
import numpy as np
from face_detector.models import face_track_server, face_describer_server
from face_detector.configs import configs
from face_aligner.align import align
import insightface
model = insightface.model_zoo.get_model('arcface_r100_v1')


model.prepare(ctx_id = -1)
class Detector:
    def __init__(self, *args, **kwargs):
        self.face_tracker = face_track_server.FaceTrackServer()
        self.face_describer = face_describer_server.FDServer(
            model_fp=configs.face_describer_model_fp,
            input_tensor_names=configs.face_describer_input_tensor_names,
            output_tensor_names=configs.face_describer_output_tensor_names,
            device=configs.face_describer_device,
        )

    def reg_face(self, frame):
        frame = frame[:,:,::-1]
        self.face_tracker.process_lm(frame)
        _landmarks= self.face_tracker.get_face_landmarks()
        _face_description = []
        if len(_landmarks) > 0:
            _face_resize = align(frame,_landmarks)
            _data_feed = [
                np.expand_dims(_face_resize, axis=0),
                configs.face_describer_drop_out_rate,
            ]
            _face_description = self.face_describer.inference(_data_feed)[0][0]

        return _face_description

    def reg_face_mx(self,frame):
        self.face_tracker.process_lm(frame)
        _landmarks= self.face_tracker.get_face_landmarks()
        _face_description = []
        if len(_landmarks) > 0:
            _face_resize = align(frame,_landmarks)
            cv2.imwrite("face.png",_face_resize)
            # cv2.waitKey(0)
            # _face_resize = cv2.resize(face, configs.face_describer_tensor_shape)
            # _data_feed = [
            #     np.expand_dims(_face_resize, axis=0),
            #     configs.face_describer_drop_out_rate,
            # ]
            # _face_description = self.face_describer.inference(_data_feed)[0][0]
            _face_description = model.get_embedding(_face_resize)
        return _face_description