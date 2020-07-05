<h1 align="center">COVID19 Face Mask Detection with Recognition for College</h1>

<div align= "center">
  <h4>Face Mask Detection & Recognition system built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images as well as in real-time video streams.</h4>
</div>

## :innocent: Motivation
In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of __‘with_mask’__ images has made this task more cumbersome and challenging. The face recognizer application which can identify the face(s) of the person(s) showing on a web cam is inspired by two path breaking papers on facial recognition using deep convoluted neural network, namely FaceNet and DeepFace.I have used pre trained model Keras-OpenFace which is an open source Keras implementation of the OpenFace (Originally Torch implemented).
 
## :hourglass: Project Demo

Face Mask Detetction on a live video 👇👇👇👇👇👇

<p align="center"><img src="https://github.com/vedantsahai18/College-Covid-System/blob/master/images/Demo.gif" width="720" height="480"></p>

Face Mask Detetction on an input image 👇👇👇👇👇👇

<p align="center"><img src="https://github.com/vedantsahai18/College-Covid-System/blob/master/images/validation.png" width="720" height="480"></p>

Face Mask Detetction and Recognition on a live video 👇👇👇👇👇👇

<p align="center"><img src="https://github.com/vedantsahai18/College-Covid-System/blob/master/images/Demo2.gif" width="720" height="480"></p>


## :warning: Tech/framework used

- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Keras-Opencface](https://github.com/iwantooxxoox/Keras-OpenFace)
- [Opencface](https://github.com/cmusatyalab/openface)
- [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)

## :star: Features
Our face mask detector didn't uses any morphed masked images dataset. The model is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

As far as the facial recognotion part is concerned , I have used Keras-OpenFace pre-trained model for feeding the face images to generate 128 dimensions embedding vector. OpenFace, which is an open face deep learning facial recognition model. It’s based on the paper: FaceNet: A Unified Embedding for Face Recognition and Clustering by Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google. OpenFace is implemented using Python and Torch which allows the network to be executed on a CPU or with CUDA.

Open Face Architecture 👇👇👇👇👇👇

![](https://github.com/vedantsahai18/College-Covid-System/blob/master/images/openface.png)

I am using this pre-trained network to compare the embedding vectors of the images stored in the file system with the embedding vector of the image captured from the webcam. This can be explained by the below diagram.

![](https://github.com/vedantsahai18/College-Covid-System/blob/master/images/faceuseonshot.png)


As per the above diagram, if the face captured by webcam has similar 128-bit embedding vector stored in the database then it can recognize the person. All the images stored in the file system are converted to a dictionary with names as key and embedding vectors as value.

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

# One Shot Learning
In one shot learning, only one image per person is stored in the database which is passed through the neural network to generate an embedding vector. This embedding vector is compared with the vector generated for the person who has to be recognized. If there exist similarities between the two vectors then the system recognizes that person, else that person is not there in the database. This can be understood by below picture.

![](https://github.com/vedantsahai18/College-Covid-System/blob/master/images/One%20Shot%20Learning.JPG)

# Triplet Loss Function
Here we are using OpenFace pre-trained model for facial recognition. Without going into much details on how this neural network identify two same faces, let's say that the model is trained on a large set of face data with a loss function which groups identical images together and separate non-identical faces away from each other. Its also known as triplet loss function.

![](https://github.com/vedantsahai18/College-Covid-System/blob/master/images/Triplet%20Loss%20Function.JPG)

## :file_folder: Dataset

Please download the dataset in the dataset folder and unzip inside it .The link is as follows- [Click to Download](https://drive.google.com/drive/folders/1o3L0lNbhU3Vq8HjweHTT6VINbEL7ejuj?usp=sharing)

This dataset consists of __10563 images__ belonging to two classes:
*	__with_mask: 7130 images__
*	__without_mask: 3433 images__

The images used were real images of faces wearing masks. The images were collected from the following sources:

* __Kaggle datasets__ 
* __RMFD dataset__ ([See here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset))

## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/vedantsahai18/College-Covid-System/blob/master/requirements.txt)

## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/vedantsahai18/COVID-Face-Mask-Detection.git
```

2. Change your directory to the cloned repo and create a Python virtual environment named 'test'
```
$ mkvirtualenv test
```

3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory folder and type the following command:
```
$ python train_mask_detector.py --dataset dataset
```

2. To train facial recognition 
```
$ python detect_mask_image.py --image images/pic1.jpeg
```

3. To train mask detection 
```
$ python train_mask_detector.py --datatset dataset
```

4. Now detect the face masks in images 
```
$ python detect_mask_image.py --image images/pic1.jpeg
```

5. To capture image of yourself ( type name and press enter and then exit/next person) 
```
$ python capture.py
```

6. Detection & Recognition in real-time video streams
```
$ python detect_mask_video.py 
```
## :key: Results

#### Our model gave 93% accuracy for Face Mask Detection after training via <code>tensorflow-gpu==2.0.0</code>

![](https://github.com/vedantsahai18/College-Covid-System/blob/master/images/evaluate.PNG)

#### We got the following accuracy/loss training curve plot
![](https://github.com/vedantsahai18/College-Covid-System/blob/master/images/plot.png)

## 📜📜 TODO

* Gather actual images (rather than artificially generated images) of people wearing masks.

* Gather images of faces that may “confuse” our classifier into thinking the person is wearing a mask when in fact they are not.

* Consider training a dedicated two-class object detector rather than a simple image classifier.


## :heart: Owner
Made with :heart:&nbsp;  by [ Vedant Sahai](https://github.com/vedantsahai18)
