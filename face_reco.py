import cv2
import face_recognition
import os
import numpy as np

# Load the face and eye cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Directory containing images of known faces
known_faces_dir = 'pic'

# Load faces
known_face_encodings = []
known_face_names = []
for file_name in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, file_name)
    known_image = face_recognition.load_image_file(img_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_face_encodings.append(known_encoding)
    known_face_names.append(file_name.split('.')[0])

# webcam
cap = cv2.VideoCapture(0)

# eye size changes
prev_eye_size = None
min_eye_size_change = 5  # Minimum change in eye size to detect

# Variables for detecting fake frames
consecutive_fake_frames = 0
consecutive_fake_frames_threshold = 20  # Number of frames to declare as fake

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Cheack if eyes are detected
        if len(eyes) > 0:
            # Calculate the average eyes size
            eye_sizes = [eye[2] for eye in eyes]
            avg_eye_size = sum(eye_sizes) / len(eye_sizes)

            # Cheack for a significant change in eye size
            if prev_eye_size is not None and abs(avg_eye_size - prev_eye_size) > min_eye_size_change:
                # Reset consecutive fake frames counter if a significant change is detected
                consecutive_fake_frames = 0
                print("Real Person")
            else:
                # Increment fake frames counter
                consecutive_fake_frames += 1
                if consecutive_fake_frames >= consecutive_fake_frames_threshold:
                    print("Fake Person Detected!")

            # Update the previous eye size
            prev_eye_size = avg_eye_size

        else:
            # If no eyes are detected, reset the fake frames counter
            consecutive_fake_frames = 0
            print("No Eyes Detected")

        # Use face recognition to check if the face is a known face
        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangles around the face and eyes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Display the name of the recognized face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()