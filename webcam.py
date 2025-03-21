# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from detect_identity import create_input_image_embeddings,recognize_face
from detect_mask import detect_and_predict_mask

screenWidth = 1200
screenHeight = 700
identity= None

def predict_webcam():
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

	# # load the face mask detector model from disk
	# print("[INFO] loading face detection model...")
	# facerec = load_model('./model/face-rec.model')

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=700)
		(h, w) = frame.shape[:2]
		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,  args["confidence"])
		input_embeddings,facerec = create_input_image_embeddings()
		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
			(startX1, startY1) = (max(0, startX), max(0, startY))
			(endX2, endY2) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
			face = frame[startY1:endY2, startX1:endX2]
			identity = recognize_face(face, input_embeddings, facerec)
			print("++++",identity)
			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			label = label + ' ' + str(identity)
			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# show the output frame
		frame2=frame[0:screenHeight,0:screenWidth]
		cv2.imshow("Frame", frame2)
		
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

predict_webcam()