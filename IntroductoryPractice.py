import face_recognition
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import time
import dlib
import cv2
from pynput.keyboard import Key, Listener
import keyboard
import os
import tkinter as tk
# Code to add widgets will go here...


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
TOTAL = 0
count = 0
faceRec = 1


vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)
fps = FPS().start()
cv2.namedWindow("Smile Detector")
shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
COUNTER = 0
known_face_encodings = []

def smile(mouth):
    d = dist.euclidean(mouth[0], mouth[6])
    l = (dist.euclidean(mouth[2], mouth[10]) + dist.euclidean(mouth[3], mouth[9]) + dist.euclidean(mouth[4], mouth[8]))/(3*d)
    return l




str = "Barack.jpeg"
# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file(str)
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("Brad Pitt.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Brad Pitt"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



while True:
    if cv2.waitKey(1) & 0xFF == ord('s'):
        faceRec *= -1

    if(faceRec == 1):
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)


        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if (name == "Unknown" and count % 10 == 0 and count > 0):
                TOTAL += 1
                print("yes")

                nName = input("Name: ")
                img_name = "{}.png".format(nName)
                img = cv2.imwrite(img_name, frame)
                new_image = face_recognition.load_image_file(nName + ".png")

                new_image_encoding = face_recognition.face_encodings(new_image)[0]
                known_face_encodings.append(new_image_encoding)
                known_face_names.append(nName)
                count = 0
            elif(name == "Unknown"):
                count += 1
        #print(count)


    elif(faceRec == -1):
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        # will need to code here

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            # print(shape)
            #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            mar = smile(mouth)
            print(mar)
            if mar > 0.37:
                COUNTER += 1

                if COUNTER >= 10:
                    TOTAL += 1
                    time.sleep(2)
                    img_name = "opencv_frame_{}.png".format(TOTAL)
                    cv2.imwrite(img_name, frame)
                    COUNTER = 0
                    im = cv2.imread(img_name)
                    r = cv2.selectROI('select',im)
                    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                    img2_name = "opencv_crop_{}.png".format(TOTAL)
                    cv2.imwrite(img2_name, imCrop)
                    cv2.destroyWindow('select')
                    time.sleep(2)
            #cv2.putText(frame, "Smile Detect {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #print(count)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
