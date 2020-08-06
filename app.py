import flask
from flask import render_template, jsonify, request, Flask,Response
import requests
import json
import urllib
import cv2
from face_detector.detector import Detector
import numpy as np 

detector = Detector()
app = Flask(__name__)


@app.route("/")
def index():
    data = {"success":True}
    return data


@app.route("/detect",methods=['POST'])
def detect():
    if flask.request.method == "POST":
        img_url = request.values.get("img_url")
        image=url_to_img(img_url)
        embedding = detector.reg_face(image)
        data={"embedding": str(embedding)}
        return data,200

def url_to_img(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img



if __name__ == "__main__":
    app.run(host='0.0.0.0',port='80')