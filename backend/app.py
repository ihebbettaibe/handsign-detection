import json
import base64
import io
import logging
import pickle
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model.") from e

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Labels dictionary
labels_dict = {0: 'T', 1: 'H', 2: 'I', 3: 'S', 4: 'A', 5: 'D', 6: 'E', 7: 'M', 8: 'O'}

# Preprocess input image and extract features
def preprocess_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        results = hands.process(image_np)
        if not results.multi_hand_landmarks:
            logger.warning("No hands detected in the image.")
            return None

        # Use only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        x_vals = [landmark.x for landmark in hand_landmarks.landmark]
        y_vals = [landmark.y for landmark in hand_landmarks.landmark]
        min_x = min(x_vals)
        min_y = min(y_vals)

        data_aux = []
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min_x)
            data_aux.append(landmark.y - min_y)

        if len(data_aux) != 42:
            logger.warning(f"Invalid feature vector length: {len(data_aux)}")
            return None

        return np.array(data_aux)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                
                if "frame" not in json_data:
                    continue

                frame_data = json_data.get("frame")
                image_data = base64.b64decode(frame_data)
                features = preprocess_image(image_data)

                if features is None:
                    await websocket.send_json({"prediction": "No hands detected"})
                else:
                    prediction = model.predict([features])
                    predicted_character = labels_dict[int(prediction[0])]
                    print(f"Predicted: {predicted_character}")  # Debug log
                    await websocket.send_json({"prediction": predicted_character})

            except Exception as e:
                logger.error(f"Error: {str(e)}")
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        features = preprocess_image(image_data)

        if features is None:
            raise HTTPException(status_code=400, detail="No hands detected or invalid input")

        prediction = model.predict([features])
        predicted_character = labels_dict[int(prediction[0])]

        return {"predicted_character": predicted_character}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
