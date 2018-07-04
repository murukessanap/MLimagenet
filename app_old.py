# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
#from keras import imagenet_utils
from keras.preprocessing import image as image_utils
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
#from imagenet_utils import decode_predictions
#from imagenet_utils import preprocess_input
#from vgg16 import VGG16
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import argparse
import cv2

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

@app.route("/test", methods=["POST"])
def test(image):
	# construct the argument parse and parse the arguments
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--image", required=True,
	#	help="path to the input image")
	#args = vars(ap.parse_args())

	# load the original image via OpenCV so we can draw on it and display
	# it to our screen later
	#orig = cv2.imread(args["image"])

	# load the input image using the Keras helper utility while ensuring
	# that the image is resized to 224x224 pxiels, the required input
	# dimensions for the network -- then convert the PIL image to a
	# NumPy array
	#print("[INFO] loading and preprocessing image...")
	#image = image_utils.load_img(args["image"], target_size=(224, 224))
	#image = image_utils.img_to_array(image)

	# our image is now represented by a NumPy array of shape (3, 224, 224),
	# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
	# pass it through the network -- we'll also preprocess the image by
	# subtracting the mean RGB pixel intensity from the ImageNet dataset
	#image = np.expand_dims(image, axis=0)
	#image = preprocess_input(image)

	# load the VGG16 network
	print("[INFO] loading network...")
	#model = VGG16(weights="imagenet")

	# classify the image
	print("[INFO] classifying image...")
	preds = model.predict(image)
	(inID, label) = decode_predictions(preds)[0]

	# display the predictions to our screen
	print("ImageNet ID: {}, Label: {}".format(inID, label))

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	#model = ResNet50(weights="imagenet")
	model = VGG16(weights="imagenet")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))
			test(image)
			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()