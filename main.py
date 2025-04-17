import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError

# Constants
IMAGE_SIZE = 48
AGE_GENDER_IMAGE_SIZE = 128
CLASS_NAMES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_PATH = './models/'  # Path where your models are saved

# Create model directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Function to load the models
def load_models():
    custom_objects = {'mae': MeanAbsoluteError()}
    try:
        # Load pre-trained models with custom objects
        emotion_model = load_model(os.path.join(MODEL_PATH, 'emotion_model.h5'))
        age_model = load_model(os.path.join(MODEL_PATH, 'age_model.h5'), custom_objects=custom_objects)
        gender_model = load_model(os.path.join(MODEL_PATH, 'gender_model.h5'))
        print("Models loaded successfully!")
        return emotion_model, age_model, gender_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Function to predict emotion, age, and gender
def predict(emotion_model, age_model, gender_model, roi_gray, roi_color):
    # Emotion prediction
    roi_gray_emotion = cv2.resize(roi_gray, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    roi_gray_emotion = roi_gray_emotion.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    emotion_pred = emotion_model.predict(roi_gray_emotion, verbose=0)
    emotion_probabilities = emotion_pred[0]

    # Age prediction
    roi_color_age = cv2.resize(roi_color, (AGE_GENDER_IMAGE_SIZE, AGE_GENDER_IMAGE_SIZE)) / 255.0
    roi_color_age = roi_color_age.reshape(1, AGE_GENDER_IMAGE_SIZE, AGE_GENDER_IMAGE_SIZE, 3)
    age_pred = age_model.predict(roi_color_age, verbose=0)
    predicted_age = int(age_pred[0][0])

    # Gender prediction
    roi_color_gender = cv2.resize(roi_color, (AGE_GENDER_IMAGE_SIZE, AGE_GENDER_IMAGE_SIZE)) / 255.0
    roi_color_gender = roi_color_gender.reshape(1, AGE_GENDER_IMAGE_SIZE, AGE_GENDER_IMAGE_SIZE, 3)
    gender_pred = gender_model.predict(roi_color_gender, verbose=0)
    gender_confidence = float(gender_pred[0][0])
    predicted_gender = 'Male' if gender_confidence < 0.5 else 'Female'

    return emotion_probabilities, predicted_age, predicted_gender, gender_confidence

# Function to draw text with background
def draw_text_with_bg(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                      text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    bg_x1, bg_y1 = x - 5, y - text_size[1] - 5
    bg_x2, bg_y2 = x + text_size[0] + 5, y + 5
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

def main():
    # Load models
    emotion_model, age_model, gender_model = load_models()
    if emotion_model is None or age_model is None or gender_model is None:
        print("Could not load all required models. Exiting...")
        return

    # Initialize webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return

    # Set camera resolution (if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Webcam is now active. Controls:")
    print("- Press 'v' to toggle confidence display")
    print("- Press 'q' to quit")

    show_confidence = False

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image")
            break

        # Create a copy of the frame for display
        display_frame = frame.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Sort faces by position (left to right)
        faces = sorted(faces, key=lambda x: x[0])

        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around the face
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            try:
                emotion_probabilities, age, gender, gender_conf = predict(
                    emotion_model, age_model, gender_model, roi_gray, roi_color
                )

                # Determine the dominant emotion
                dominant_emotion_index = np.argmax(emotion_probabilities)
                dominant_emotion = CLASS_NAMES[dominant_emotion_index]
                dominant_emotion_prob = emotion_probabilities[dominant_emotion_index] * 100

                # Display dominant emotion above the face box
                draw_text_with_bg(display_frame, f"{dominant_emotion}: {dominant_emotion_prob:.1f}%",
                                  (x, y - 10), bg_color=(0, 0, 255))

                # Display emotion percentages to the left of the face box
                text_x_offset = x - 150  # Start position to the left of the face box
                for j, emotion in enumerate(CLASS_NAMES):
                    emotion_text = f"{emotion}: {emotion_probabilities[j] * 100:.1f}%"
                    draw_text_with_bg(display_frame, emotion_text, (text_x_offset, y + 20 * j),
                                      bg_color=(0, 0, 0))

                # Display age and gender below the emotion percentages
                draw_text_with_bg(display_frame, f"Age: {age}", (text_x_offset, y + 20 * len(CLASS_NAMES)),
                                  bg_color=(50, 50, 50))
                draw_text_with_bg(display_frame, gender, (text_x_offset, y + 20 * (len(CLASS_NAMES) + 1)),
                                  bg_color=(100, 100, 100))

                # Display confidence scores if enabled
                if show_confidence:
                    gender_text = f"Conf: {gender_conf:.2f}"
                    draw_text_with_bg(display_frame, gender_text, (text_x_offset, y + 20 * (len(CLASS_NAMES) + 2)),
                                      bg_color=(50, 50, 150))

            except Exception as e:
                print(f"Prediction error: {e}")

        # Add status indicators
        conf_text = "Confidence: ON" if show_confidence else "Confidence: OFF"
        draw_text_with_bg(display_frame, conf_text, (10, 30), bg_color=(0, 50, 0))
        draw_text_with_bg(display_frame, "Press 'q' to quit", (10, 60), bg_color=(0, 0, 50))

        # Display the frame
        cv2.imshow('Emotion, Age and Gender Detection', display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('v'):  # Toggle confidence display
            show_confidence = not show_confidence
            print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU available and configured")
    else:
        print("Running on CPU")

    main()
