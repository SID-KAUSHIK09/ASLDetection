from flask import Flask,render_template
import numpy as np
import cv2
import wx
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
from cvzone.ClassificationModule import Classifier
import time
import winsound

app = Flask(__name__)


def asl_():
    offset = 20
    imgSize = 300
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '  ', 'thanks', 'hello']
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("C:/Users/91911/Desktop/Minor2_Final/sid_model/keras_model.h5", "C:/Users/91911/Desktop/Minor2_Final/sid_model/labels.txt")

    file = open("C:/Users/91911/Desktop/Minor2_Final/recognized_characters.txt", "w")
    start_time = time.time()

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wGap+wCal] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, labels[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                k = imgSize/w
                hCal = math.ceil(h*k)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hGap+hCal, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, labels[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if time.time() - start_time > 5:
                print("Predicted character:", labels[index])

                file.write(labels[index])  # Write recognized character to file
                file.flush()  
                start_time = time.time()
                winsound.PlaySound("SystemExit", winsound.SND_ALIAS)  # Play sound when character is written

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    file.close()  # Close file object to save and close the text file
    cv2.destroyAllWindows()
    cap.release()
    return render_template('index.html')

@app.route('/')
def hello_world():
    return render_template('index.html')



@app.route('/asl',methods=['Get','Post'])
def asl():
    asl_()
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
