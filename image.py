# USAGE
# python image.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import imutils
import os
from detect_identity import create_input_image_embeddings,recognize_face
from detect_mask import detect_and_predict_mask

def image(url):
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", type=str,
		default="model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector caffe...")
	prototxtPath = os.path.sep.join([args["model"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["model"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	modelpath = os.path.sep.join([args["model"], "mask_detector.model"])
	maskNet = load_model(modelpath)

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(url)
	orig = image.copy()
	(h, w) = image.shape[:2]

	(locs, preds) = detect_and_predict_mask(image, faceNet, maskNet,  args["confidence"])
	input_embeddings,facerec = create_input_image_embeddings()	

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
			# detect the face , resize the box and recognize it 
		(startX1, startY1) = (max(0, startX), max(0, startY))
		(endX2, endY2) = (min(w - 1, endX), min(h - 1, endY))
		face_rec = image[startY1:endY2, startX1:endX2]
		identity = recognize_face(face_rec, input_embeddings, facerec)
			# determine the class label and color we'll use to draw
			# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		label = label + ' ' + str(identity)
			# display the label and bounding box rectangle on the output
			# frame
		cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
			# show the output image
		cv2.imshow("Output", image)
		cv2.waitKey(0)
			#cv2.destroyAllWindows()

image('./face_images/Vedant.jpg')
