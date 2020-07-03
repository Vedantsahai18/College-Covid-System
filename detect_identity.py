from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import utils
import glob
from utils import LRN2D

# with CustomObjectScope({'tf': tf}):
#     model = load_model('./model/face-rec.h5')

# About image_to_embedding function
# When the model is loaded with pre trained weights, then we can create the 128 dimensional embedding vectors for all the face images stored in the "images" folder. 
# "image_to_embedding" function pass an image to the Inception network to generate the embedding vector.

def image_to_embedding(image, model):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA) 
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

# About recognize_face function
# This function calculate similarity between the captured image and the images that are already been stored. It passes the image to the trained neural network to generate its embedding vector. 
# Which is then compared with all the embedding vectors of the images stored by calculating L2 Euclidean distance.
# If the minimum L2 distance between two embeddings is less than a threshold (here I have taken the threashhold as .68 (which can be adjusted) then we have a match.

def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)
    
    minimum_distance = 200
    name = None
    
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():
        
       
        euclidean_distance = np.linalg.norm(embedding-input_embedding)
        

        print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))

        
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance < 1.5:
        return str(name)
    else:
        return None

# About create_input_image_embeddings function
# This function generates 128 dimensional image ebeddings of all the images stored in the "images" directory by feed forwarding the images to a trained neural network. 
# It creates a dictionary with key as the name of the face and value as embedding

# About recognize_faces_in_cam function
# This function capture image from the webcam, detect a face in it and crop the image to have a face only, which is then passed to recognize_face function.

def create_input_image_embeddings():
    input_embeddings = {}
    with CustomObjectScope({'tf': tf}):
        model = load_model('./model/face-rec.h5')
    for file in glob.glob("face_images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)

    return input_embeddings,model
