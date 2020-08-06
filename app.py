import flask
from flask import render_template, jsonify, request, Flask,Response
import requests
import json
import urllib
import cv2
from face_detector.detector import Detector
import numpy as np 
from urllib.parse import unquote
from cryptography.fernet import Fernet

detector = Detector()
app = Flask(__name__)


@app.route("/")
def index():
    data = {"success":True}
    return data

@app.route("/detectnormal",methods=["POST"])
def detect_normal():
    if flask.request.method == "POST":
        img_url = request.values.get("img_url")
        image = url_to_img(img_url)
        embedding = detector.reg_face(image)
        data = {"embedding":str(embedding)}
        return data,200


@app.route("/detect",methods=['POST'])
def detect():
    if flask.request.method == "POST":
        encrypted_message = request.values.get("img_url")
        img_url=decrypt_message(encrypted_message.encode())
        image=url_to_img(img_url)
        embedding = detector.reg_face(image)
        data={"embedding": str(embedding)}
        return data,200

def url_to_img(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img

def load_key():
    return open("secret.key", "rb").read()

def decrypt_message(encrypted_message):
    key = load_key()
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    decrypted_message = decrypted_message.decode("utf-8")
    return decrypted_message

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='80')