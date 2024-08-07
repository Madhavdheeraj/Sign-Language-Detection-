# Real-Time Two-Way Sign Language Communication System

## Click on the below link for explaination
[Watch the video](https://www.youtube.com/watch?v=88Ff9QOND_Q)

## Description
The Real-Time Two-Way Sign Language Communication System aims to address the communication barrier experienced by the Deaf and Hard of Hearing (DHH) community when interacting with individuals who do not understand sign language. This project focuses on developing a comprehensive real-time translation tool that seamlessly converts American Sign Language (ASL) to English and vice versa. The system operates in two modes:
- *Video Call Translation*: Integrated into a video call platform for face-to-face interactions.
- *Camera and Microphone Translation*: A mobile application that uses the device’s camera and microphone to recognize and translate sign language to English and spoken English sentences to ASL.

## Problem Statement
The DHH community faces challenges in accessing critical services, education, healthcare, and employment opportunities due to communication barriers. Existing video call interpretation services are costly, subject to availability constraints, and may compromise user privacy.

## Goals
Our goal is to develop a comprehensive real-time two-way communication model that:
- Converts ASL to English and vice versa in real-time.
- Operates seamlessly in video calls and camera-based interactions.
- Utilizes cutting-edge technologies including computer vision, natural language processing (NLP), machine learning, and deep learning.

## Key Features and Technologies
### Video Call Translation:
- *Computer Vision*: Recognizes and interprets sign language gestures during video calls.
- *Natural Language Processing*: Converts sign language text into spoken language.
- *Machine Learning*: Trains models using datasets to enhance translation accuracy.
- *End-to-end Encryption*: Ensures the security and privacy of video call translations.

### Camera-Based Translation:
- Utilizes the same technologies as video call translation for camera-based interactions.

## Tools Used
- OpenCV
- HandTrackingModule
- Classifier (CNN Model)
- TensorFlow/Keras

## Procedure

## Data Collection
To train the deep learning model for hand gesture recognition, a dataset of hand gesture images labeled with corresponding classes (e.g., letters of the alphabet in sign language) was collected. The dataset should include various hand gestures to ensure robustness and accuracy of the model.

![Hand Gesture Dataset](IMages/Data_collection.png)
![Hand Gesture Dataset](IMages/data_collection1.jpg)

### Model Training
- Train the deep learning model using Convolutional Neural Networks (CNN) for hand gesture recognition.
- Collect a dataset of hand gesture images labeled with corresponding classes (e.g., letters of the alphabet in sign language).
- Preprocess the images by resizing, normalization, and augmentation as necessary.
- Define and compile a CNN architecture using popular deep learning frameworks like TensorFlow or Keras.
- Train the CNN model on the preprocessed dataset using appropriate training parameters (e.g., batch size, number of epochs).
- Validate the trained model on a separate validation dataset to assess its performance and generalization ability.
- Fine-tune the model and adjust hyperparameters based on validation results to improve accuracy and convergence.
- Once satisfied with the model's performance, save the trained model to an h5 file using the model.save() method in Keras.

### Output of using trained model
![Hand Gesture Dataset](IMages/Output.png)

### Server Setup
- Run the server.py code on a dedicated server or a host machine with a static IP address.
- Define the host IP address (HOST) and port (PORT) in the server code.
- The server listens for incoming connections from clients and handles data transmission.

### Client 1 Setup
- Run the client1.py code on the device where input (hand gestures) will be captured.
- Adjust the SERVER_HOST and SERVER_PORT variables in the client code to match the server's IP address and port.
- Connect client1.py to the server by establishing a socket connection.
- Capture live video streams from the camera using OpenCV.
- Utilize the HandTrackingModule to detect and track hand gestures in the video stream.
- Classify the detected hand gestures using the Classifier module (CNN Model).
- Send the recognized hand gestures to the server for further processing and transmission to client2.py.

![Hand Gesture Dataset](IMages/B.png)
![Hand Gesture Dataset](IMages/E.png)
![Hand Gesture Dataset](IMages/D.png)

![Hand Gesture Dataset](IMages/client1.png)


### Client 2 Setup
- Run the client2.py code on the device where the received input will be stored.
- Similar to client1.py, configure the SERVER_HOST and SERVER_PORT variables to connect to the server.
- Establish a socket connection with the server to receive data transmitted by client1.py.
- Receive and store the input (recognized hand gestures) sent by client1.py from the server.
- Display the received input or perform any further actions as required by the application.

![Hand Gesture Dataset](IMages/client2.png)

### Real-Time Communication
- Ensure that both client1.py and client2.py are connected to the server simultaneously.
- client1.py continuously captures hand gestures, classifies them, and sends the recognized gestures to the server in real-time.
- The server receives the data from client1.py, processes it if necessary, and forwards it to client2.py.
- client2.py receives the data from the server, stores it, and performs any required actions based on the received input.

![Hand Gesture Dataset](IMages/Server.png)

## Conclusion
Please note that the current repository demonstrates a simplified client-server interaction for two-way communication. The final implementation with WebRTC offers a more comprehensive solution, enabling seamless real-time video call translation with enhanced accessibility features.

## Future Implementation with WebRTC
In further development we will be implementing this in a real time video call platform where the deaf and dumb person and a normal person can interact.

## Demo
<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <p>Before</p>
    <img src="IMages/1.png" alt="Alt Text 1" style="width: 400px; height: 300px; margin: 10px;">
  </div>
  <div style="text-align: center;">
    <p>After</p>
    <img src="IMages/2.png" alt="Alt Text 2" style="width: 400px; height: 300px; margin: 10px;">
  </div>
</div>

## Installations
- Python 3.x
- OpenCV
- TensorFlow/Keras
- cvzone

