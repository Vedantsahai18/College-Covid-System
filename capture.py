import face_recognition
import cv2
import csv
import numpy as np
from datetime import datetime

# Initialise cam, window name, img_counter and users
cam = cv2.VideoCapture(1)
cv2.namedWindow("test")
img_counter = 1
userInput = [" Hello to:", " "]
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
            # Draw a box around the face
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)

    k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video

    if k == 27:
        break

    # NAME RECORDING SECTION
    if ret == True:
        # draw the label into the frame
        userInputLen = len(userInput)
        # print("drawing")
        for i in range(userInputLen):
            __draw_label(frame, userInput[i], (20,50+ 40*i), (255,0,0))
    
    
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
        cv2.imwrite(img_path, frame) 
        print("{} written!".format(img_name))
        img_counter += 1

        userInput.append(" ")

    # Display the resulting image
    cv2.imshow('Video', frame)


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()