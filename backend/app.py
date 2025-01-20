from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image
import io
import logging
import json  # To handle JSON parsing
import base64  # To handle base64 encoding/decoding

# Initialize FastAPI
app = FastAPI()

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

        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        if len(data_aux) != 42:  # Replace with your model's expected feature size if different
            logger.warning(f"Invalid feature vector length: {len(data_aux)}")
            return None

        return np.array(data_aux)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data).get("frame")

            # Decode base64 frame and preprocess
            image_data = base64.b64decode(frame_data)
            features = preprocess_image(image_data)

            if features is None:
                response = {"prediction": "No hands detected or invalid input"}
            else:
                prediction = model.predict([features])
                predicted_character = labels_dict[int(prediction[0])]
                response = {"prediction": predicted_character}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")
        await websocket.close()

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
