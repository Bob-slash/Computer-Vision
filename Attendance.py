import face_recognition
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
import datetime
import calendar


video_capture = cv2.VideoCapture(0)
TOTAL = 0
count = 0
faceRec = 1


vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)
fps = FPS().start()
cv2.namedWindow("Attendance")
#shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(shape_predictor)
#(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
check = cv2.imread('gCheck.jpeg')
#check = cv2.resize(check, ())

COUNTER = 0
known_face_encodings = []

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
known_face_encodings = []

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        now = datetime.datetime.now()
        curDate = now.strftime("%D")
        mostRecent = ""
        for line in myDataList:
            if name in line:
                entry = line.split(',')
                mostRecent = entry[1]

        if mostRecent != curDate:
            dtString = now.strftime('%D,%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def isHere(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        now = datetime.datetime.now()
        curDate = now.strftime("%D")
        mostRecent = ""
        for line in myDataList:
            if name in line:
                entry = line.split(',')
                mostRecent = entry[1]

        if mostRecent == curDate:
            return True
    return False


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#for names in classNames:


str = "Images/Bobby2.png"
# Load a sample picture and learn how to recognize it.
me_image = face_recognition.load_image_file(str)
me_face_encoding = face_recognition.face_encodings(me_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    me_face_encoding
]
known_face_names = [
    "Bobby"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


#imgTimmy = face_recognition.load_image_file('Timmy.jpeg')
#imgTimmy = cv2.cvtColor(imgTimmy.cv2.COLOR_BGR2RGB)

#cv2.imshow('Timmy', imgTimmy)

name = "Unknown"
while True:
    #if isHere(name):
    #    faceRec *= -1

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

                if name != "Unknown":
                    markAttendance(name)


        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #if(not isHere(name)):
            #    cv2.imshow('Attendance',check)
            #    cv2.waitKey(3000)
        #print(count)


    #elif(faceRec == -1):

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



