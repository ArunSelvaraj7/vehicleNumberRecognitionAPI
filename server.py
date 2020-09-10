from flask import Flask,render_template,jsonify,request,url_for,redirect
import os
from werkzeug.utils import secure_filename
import cv2
from tensorflow import keras as k
from detect import detect_plate


app = Flask(__name__)


SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
modelConfiguration = r'darknet-yolo/darknet-yolov3.cfg'
modelWeights = r'darknet-yolo/model.weights'



net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

charModel = k.models.load_model(r'charRecognition/trained_model.h5')
UPLOAD_FOLDER = r'static/images'


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanDir():
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER,file))

@app.route('/detect', methods=['POST'])
def upload_file():
    cleanDir()
	# check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        output = detect_plate(net, charModel, filename)
        if output == '':
            resp = jsonify({'message' : 'File successfully uploaded', 'Output' : 'No Number plate found!!!'})
            resp.status_code = 202
        else:
            resp = jsonify({'message' : 'File successfully uploaded', 'recognised Number' : output})
            resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
        resp.status_code = 400
        return resp

if __name__ == '__main__':
    app.run(port='5002')
