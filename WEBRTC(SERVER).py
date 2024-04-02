import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify ,redirect, url_for
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    video_file.save('uploads/uploaded_video.mp4')

    process_video()

    return redirect(url_for('output'))

@app.route('/output')
def output():
    return render_template('output.html', video_file='static/output_video.mp4')

def process_video():
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    classifier = Classifier("keras_model.h5", "labels.txt")
    offset = 20
    imgSize = 300
    labels = ["A", "B", "C", "D", "E", "F"]

    video_file = "uploads/uploaded_video.mp4" 
    cap = cv2.VideoCapture(video_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('static/output_video.mp4', fourcc, 25.0, (640, 480))  # Change resolution as needed

    while True:
        success, img = cap.read()
        
        if not success:
            break
        
        img = cv2.flip(img, 1)
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 225
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255),
                        cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        out.write(imgOutput)  

    cap.release()
    out.release()
    pass

if __name__ == '__main__':
    app.run(debug=True)
