import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the pre-trained emotion recognition model
model = load_model('my_model_v2.h5')

# Define emotions based on the model's training dataset
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize MediaPipe Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess the frame before prediction
def preprocess_frame(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
    return reshaped_face

# Function to predict emotion
def predict_emotion(face):
    processed_face = preprocess_frame(face)
    prediction = model.predict(processed_face)
    emotion_label = emotions[np.argmax(prediction)]
    return emotion_label

# Initialize video capture
cap = cv2.VideoCapture(0)

# Use MediaPipe for face detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Extract bounding box for the detected face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Crop the face for emotion prediction
                face = frame[y:y+h, x:x+w]
                if face.size != 0:  # Ensure the face crop is valid
                    emotion = predict_emotion(face)

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Draw emotion label
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Emotion Recognition with Face Landmarks", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
