import cv2
import numpy as np
import requests
import os
from face_detector.detector import Detector
import insightface
# model = insightface.model_zoo.get_model('arcface_r100_v1')


# model.prepare(ctx_id = -1)
detector = Detector()
def check(path):
    images =  os.listdir(path)
    embeddings=[]
    for image_path in images:
        image=cv2.imread(os.path.join(path,image_path))
        # embeddings.append(detector.reg_face_mx(image)[0])
        embeddings.append(detector.reg_face(image))
        # embeddings.append(model.get_embedding(cv2.resize(image,(112,112)))[0])
    print(str(len(get_similar_index(embeddings[1:],embeddings[0])))+"/"+str((len(embeddings)-1)))

def face_distance(face_encodings, face_to_compare):
    face_dist_value = []
    for encoding in face_encodings:
        face_dist_value.append(np.dot(encoding,face_to_compare)/((np.linalg.norm(encoding)*np.linalg.norm(face_to_compare))))
    # print(face_dist_value)
    return np.asarray(face_dist_value)


def get_similar_index(face_encodings,face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    indexes = np.argwhere(face_distance(face_encodings,face_to_compare) > 0.6 )
    return indexes

basepath='./VN-celeb'
entries = os.listdir("./VN-celeb")
for entry in entries:
    path=os.path.join(basepath, entry)
    # print(entry)
    check(path)

