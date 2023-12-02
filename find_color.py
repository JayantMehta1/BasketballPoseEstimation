import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture('analysis/Videos/sample_video.mp4')

# Create the color Finder object
myColorFinder = ColorFinder(True)
hsvVals = 'red'

while True:
    # Grab the image

    # success, img = cap.read()
    img = cv2.imread("analysis/colDet.png")
    img = img[0:900, :]

    # Find the Color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)


    # Display
    # imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
    cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgColor)
    cv2.waitKey(50)