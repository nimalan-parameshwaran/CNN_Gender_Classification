import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained CNN model for gender classification
model = tf.keras.models.load_model('my_cnn.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to classify gender
def classify_gender(face_image):
    # Preprocess the face image (resize, normalize, etc.)
    face_image = cv2.resize(face_image, (300, 300))  # Adjust the size as needed
    face_image = face_image / 255.0  # Normalize pixel values
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension

    # Predict gender using the model
    prediction = model.predict(face_image)
    
    if prediction[0][0] > 0.5:
        return "Male"
    else:
        return "Female"

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Classify gender
        gender = classify_gender(face)

        # Draw a rectangle around the face and display gender
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gender Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
