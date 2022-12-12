import cv2
from math import *

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
count: int = 1
prev_frame = frame

# Accessing BGR pixel values

while True:
    # Grab a single frame of video
    count = count + 1

    if count % 2 == 0:
        ret,frame = video_capture.read()
        frame = cv2.resize(frame, (0,0), fy=0.25, fx = 0.25)
        height,width, channels = frame.shape
    else:
        ret, prev_frame = video_capture.read()
        prev_frame = cv2.resize(prev_frame, (0,0), fx=0.25, fy=0.25)
        height, width, channels = prev_frame.shape

    for r in range(0, height):
        for c in range(0, width):
            curR = int(frame[r, c, 0])
            curG = int(frame[r, c, 1])
            curB = int(frame[r, c, 2])
            prevR = int(prev_frame[r,c,0])
            prevG = int(prev_frame[r, c, 1])
            prevB = int(prev_frame[r, c, 2])
            val = pow(curR-prevR, 2)+pow(curG-prevG, 2) + pow(curB-prevB, 2)
            dist = sqrt(val)
            #print(str(dist))
            if dist > 50:
                frame[r,c,0] = 255
                frame[r, c, 1] = 255
                frame[r, c, 2] = 255
            else:
                frame[r, c, 0] = 0
                frame[r, c, 1] = 0
                frame[r, c, 2] = 0

    # Display the resulting image
    if count%2 == 0:
        cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
