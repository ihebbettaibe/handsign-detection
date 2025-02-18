import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import Counter

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Labels dictionary
labels_dict = {0: 'T', 1: 'H', 2: 'I', 3: 'S', 4: 'A', 5: 'D', 6: 'E', 7: 'M', 8: 'O'}

# Stability and transition parameters
buffer_size = 10
prediction_buffer = []
gesture_stability_threshold = 8  # Minimum consistent frames for stable prediction
transition_timeout = 1.0  # Ignore new gestures for 1 second after detection
last_prediction_time = time.time()

# Initialize variables
word = ""
last_predicted_character = None
accumulated_phrase = []
max_words = 4
no_gesture_timeout = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    H, W, _ = frame.shape

    # Process only the first detected hand
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = [landmark.x for landmark in hand_landmarks.landmark]
        y_ = [landmark.y for landmark in hand_landmarks.landmark]

        # Normalize landmarks relative to the minimum x and y values
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Validate feature vector length
        if len(data_aux) != 42:
            print(f"Warning: Feature size mismatch. Expected 42, but got {len(data_aux)}")
        else:
            # Draw bounding box for visualization
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Model prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "")

            # Buffer logic for stability
            prediction_buffer.append(predicted_character)
            if len(prediction_buffer) > buffer_size:
                prediction_buffer.pop(0)
            most_common_prediction, count = Counter(prediction_buffer).most_common(1)[0]
            if count >= gesture_stability_threshold and time.time() - last_prediction_time > transition_timeout:
                if most_common_prediction != last_predicted_character:
                    word += most_common_prediction
                    last_predicted_character = most_common_prediction
                    last_prediction_time = time.time()

    # Check for no-gesture timeout regardless of hand detection
    if time.time() - last_prediction_time > no_gesture_timeout:
        if word:
            accumulated_phrase.append(word)
            word = ""
        if len(accumulated_phrase) > max_words:
            accumulated_phrase = []

    # Display current word and accumulated phrase
    cv2.putText(frame, f"Word: {word}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    phrase = " ".join(accumulated_phrase)
    cv2.putText(frame, f"Phrase: {phrase}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
