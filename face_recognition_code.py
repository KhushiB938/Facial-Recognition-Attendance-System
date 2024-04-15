# Import necessary libraries

import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
# Initialize video capture object

video_capture = cv2.VideoCapture(0)

# Load image files of known faces

khushi_image = face_recognition.load_image_file("C:/Users/Khushi/Desktop/Smart_Attendence_System/Attendance-Recording-System-Using-Facial-Recognition/photos/Khushi Bansal.png")
khushi_encoding = face_recognition.face_encodings(khushi_image)[0]

shorya_image = face_recognition.load_image_file("C:/Users/Khushi/Desktop/Smart_Attendence_System/Attendance-Recording-System-Using-Facial-Recognition/photos/Shorya Bansal.jpg")
shorya_encoding = face_recognition.face_encodings(shorya_image)[0]

vanya_image = face_recognition.load_image_file("C:/Users/Khushi/Desktop/Smart_Attendence_System/Attendance-Recording-System-Using-Facial-Recognition/photos/Vanya Goyal.jpeg")
vanya_encoding = face_recognition.face_encodings(vanya_image)[0]

dhanraj_image = face_recognition.load_image_file("C:/Users/Khushi/Desktop/Smart_Attendence_System/Attendance-Recording-System-Using-Facial-Recognition/photos/Dhanraj Bhosale.jpg")
dhanraj_encoding = face_recognition.face_encodings(dhanraj_image)[0]

ritika_image = face_recognition.load_image_file("C:/Users/Khushi/Desktop/Smart_Attendence_System/Attendance-Recording-System-Using-Facial-Recognition/photos/Ritika Kumari.jpg")
ritika_encoding = face_recognition.face_encodings(ritika_image)[0]

# Create a list of known face encodings and names

known_face_encoding = [
khushi_encoding,
shorya_encoding,
vanya_encoding,
dhanraj_encoding,
ritika_encoding
]
 
known_faces_names = [
"Khushi Bansal",
"Shorya Bansal",
"Vanya Goyal",
"Dhanraj Bhosale",
"Ritika Kumari"
]
 
# Create a copy of known faces names

students = known_faces_names.copy()

# Initialize lists to store face locations, encodings, and names

face_locations = []
face_encodings = []
face_names = []

# Set a flag to indicate whether to perform face detection and recognition

s=True

# Get current date

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
# Open a CSV file to store attendance data
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
# Start an infinite loop to continuously capture and process video frames

while True:

     # Read a frame from the video capture object

    _,frame = video_capture.read()

    # Resize the frame to reduce the computational cost of face detection and recognition

    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    # Convert the resized frame to RGB format

    rgb_small_frame = small_frame[:,:,::-1]

    # If the flag is set to True, perform face detection and recognition

    if s:

        # Detect face locations in the resized frame

        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Compute face encodings for the detected face locations

        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

        # Initialize an empty list to store face names
        face_names = []

        # Iterate over the computed face encodings
        
        for face_encoding in face_encodings:

            # Compare the computed face encoding with the known face encodings

            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""

            # Compute the face distance between the computed face encoding and the known face encodings

            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)

            # Find the index of the minimum face distance

            best_match_index = np.argmin(face_distance)

            # If the computed face encoding matches a known face encoding

            if matches[best_match_index]:

                # Get the name of the recognized person
                name = known_faces_names[best_match_index]
 
            # Add the name of the recognized person to the list of face names

            face_names.append(name)

             # If the recognized person is a student
            if name in known_faces_names:

                # Set the font properties for displaying the name of the recognized person
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 

                # Display the name of the recognized person on the frame

                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")


                    # Append the name of the recognized person to the CSV file
                    lnwriter.writerow([name,current_time])
    

     #Display the frame
    cv2.imshow("attendence system",frame)


    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 

 # Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Close the CSV file
f.close()