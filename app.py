from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from swapface import *

from PIL import Image
import base64
import io

# from script import *

app = Flask(__name__)

# Allow 
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def preprocessing_image(image_path):
	img = Image.open(image_path)
	buffer = io.BytesIO()
	img.save(buffer, 'png')
	buffer.seek(0)
	data = buffer.read()
	data = base64.b64encode(data).decode()
	return data


@app.route("/")
def hello():
	return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		print("request data", request.data)
		print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']


		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))	
	return "Success !"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	faceSwap("./input/img_1.png","./input/img_2.png")
	# object_detection(image_path=[str("./data/uploads/"+os.listdir("data/uploads/")[0])])
	faceSwap1("./input/img_2.png","./input/img_1.png")

	data = preprocessing_image("./output/"+os.listdir("./output/")[0])
	# data1 = preprocessing_image("./output/"+os.listdir("./output/")[1])
	return f'"data:image/png;base64,{data}"'


@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
	# faceSwap("./input/img_1.png","./input/img_2.png")
	# object_detection(image_path=[str("./data/uploads/"+os.listdir("data/uploads/")[0])])
	faceSwap1("./input/img_2.png","./input/img_1.png")
	data1 = preprocessing_image("./output/"+os.listdir("./output/")[1])
	return f'"data:image/png;base64,{data1}"'


if __name__ == "__main__":
	app.run(debug=True)