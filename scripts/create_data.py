import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for label_dir in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label_dir)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Couldn't read image {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # Process only the first detected hand for consistency
                hand_landmarks = results.multi_hand_landmarks[0]
                x_vals = [landmark.x for landmark in hand_landmarks.landmark]
                y_vals = [landmark.y for landmark in hand_landmarks.landmark]
                min_x = min(x_vals)
                min_y = min(y_vals)
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)
                # Ensure feature vector is the expected length (42 elements)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(label_dir)
                else:
                    print(f"Warning: Feature vector length mismatch for {img_path}")
            else:
                print(f"Warning: No hand detected in {img_path}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)