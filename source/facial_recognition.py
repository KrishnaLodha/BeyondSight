import cv2
import face_recognition
import pandas as pd
import numpy as np

# Load database from CSV
def load_faces(csv_file):
    df = pd.read_csv(csv_file)
    known_face_encodings = []
    known_face_names = []

    for index, row in df.iterrows():
        image_path = row["image_path"]
        name = row["name"]

        # Load image and encode face
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

    return known_face_encodings, known_face_names

# Load known faces from CSV file
csv_file = "faces_db.csv"
known_face_encodings, known_face_names = load_faces(csv_file)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()

    # Resize frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the first match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw bounding box around the face and display the name
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale the face location back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes and names
    cv2.imshow("Facial Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
