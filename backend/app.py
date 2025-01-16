import base64
import cv2
import numpy as np
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Correct relative path to the model file
model_path = "model.p"

# Load the trained model safely using context manager
with open(model_path, "rb") as f:
    model_dict = pickle.load(f)
model = model_dict.get("model", None)
labels_dict = {0: 'T', 1: 'H', 2: 'I', 3: 'S', 4: 'A', 5: 'D', 6: 'E', 7: 'M', 8: 'O'}

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_frame(frame):
    """
    Preprocess the frame to extract features for prediction.
    """
    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize landmarks relative to the minimum x and y
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

    return np.asarray(data_aux)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    try:
        while True:
            # Receive frame data from the client
            data = await websocket.receive_text()
            print("Received data:", data)  # Log received data
            received_data = json.loads(data)
            frame_data = received_data.get("frame", "")

            if frame_data:
                # Decode the base64 image
                frame_bytes = base64.b64decode(frame_data.split(",")[1])
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Preprocess the frame
                features = preprocess_frame(frame)

                # Check if the model is loaded correctly and features are extracted
                if model is not None and len(features) > 0:
                    # Make prediction
                    prediction = model.predict([features])
                    predicted_label = labels_dict.get(int(prediction[0]), "Unknown")

                    # Send the prediction back to the client
                    await websocket.send_text(json.dumps({"prediction": predicted_label}))
                else:
                    await websocket.send_text(json.dumps({"prediction": "Model not loaded or no hand detected"}))
            else:
                await websocket.send_text(json.dumps({"prediction": "No frame received"}))
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error processing frame: {e}")
        await websocket.send_text(json.dumps({"prediction": "Error occurred"}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6001)