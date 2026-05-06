print("🚀 Script started...")

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# 🔥 MODEL PATH (change if needed)
MODEL_PATH = "S:\new-task-46cb\new-task-46cb\backend\model\best_model.h5"
LABEL_PATH = "S:\new-task-46cb\new-task-46cb\backend\model\labels.txt"

# 🔍 Check files
print("Model exists:", os.path.exists(MODEL_PATH))
print("Labels exists:", os.path.exists(LABEL_PATH))

# Load model
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Load labels
with open(LABEL_PATH) as f:
    labels = f.read().splitlines()

print("✅ Labels:", labels)

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 20

# 🔹 Extract keypoints
def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# 🔹 Normalize
def normalize_keypoints(kp):
    return kp / (np.max(np.abs(kp)) + 1e-6)

# 🔹 Main
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not working")
        return

    print("✅ Camera opened")

    sequence = []
    output_text = ""

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Get keypoints
            keypoints = extract_keypoints(results)
            keypoints = normalize_keypoints(keypoints)

            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            # 🔥 Prediction
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)

                prediction = model.predict(input_data, verbose=0)[0]

                class_id = np.argmax(prediction)
                label = labels[class_id]
                confidence = float(prediction[class_id])

                print("RAW:", label, confidence)

                # 🔥 LOW THRESHOLD (important)
                if confidence > 0.3:
                    output_text = label

            # Display
            cv2.putText(image, output_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Sign Recognition", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()