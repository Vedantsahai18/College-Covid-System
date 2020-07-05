import face_recognition
import cv2
import csv
import numpy as np
import imutils
import argparse
import os
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
		default="model",
		help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialise cam, window name, img_counter and users
cam = cv2.VideoCapture(1)
cv2.namedWindow("test")
img_counter = 1
userInput = [" Hello to:", " "]
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# load our serialized face detector model from disk
print("[INFO] loading face detector caffe...")
prototxtPath = os.path.sep.join([args["model"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["model"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
screenWidth = 1200
screenHeight = 700

def __draw_label(img, text, pos, bg_color):
	font_face = cv2.FONT_HERSHEY_SIMPLEX
	scale = 1.3
	color = (0, 0, 255)
	thickness = cv2.FILLED *2
	margin = 2
	txt_size = cv2.getTextSize(text, font_face, scale, thickness)
	cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

vs = cv2.VideoCapture(0)

while True:
	# Grab a single frame of video
	# to have a maximum width of 400 pixels
	ret, image = vs.read()
	image = imutils.resize(image, width=700)
	(h, w) = image.shape[:2]
   # construct a blob from the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	for i in range(0, detections.shape[2]):
		
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

	k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video

	if k == 27:
		break

	# NAME RECORDING SECTION
	if ret == True:
		# draw the label into the frame
		userInputLen = len(userInput)
		# print("drawing")
		for i in range(userInputLen):
			__draw_label(image, userInput[i], (20,50+ 40*i), (255,0,0))
	
	
	for i in range(97,123):
		if k == i:
			if userInput[-1][-1] == ' ':
				userInput[-1] += chr(i-32)
			else:
				userInput[-1] += chr(i)

	if k%256 == 32:
		userInput[-1] += ' '

	elif k%256 == 13:
		# ENTER pressed
		img_name = "{}.jpg".format(userInput[-1][1:])
		img_path = "./face_images/" + img_name
		cv2.imwrite(img_path, image) 
		print("{} written!".format(img_name))
		img_counter += 1

		userInput.append(" ")

	# Display the resulting image
	cv2.imshow('Video', image)


# Release handle to the webcam
vs.release()
cv2.destroyAllWindows()