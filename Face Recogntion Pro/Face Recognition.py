import cv2 as cv

import numpy as np

# Get user supplied values
# Create the haar cascade

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the image
image = cv.imread('ffcc.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv.cv.CV_HAAR_SCALE_IMAGE
)

print ("Found {0} faces!" .format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow("Faces found", image)
cv.waitKey(0)    