const webcam = document.getElementById('webcam');
const outputCanvas = document.getElementById('outputCanvas');
const predictionText = document.getElementById('predicted-sign');
const ctx = outputCanvas.getContext('2d');

// Initialize WebSocket
const socket = new WebSocket('ws://127.0.0.1:6001/ws');

// Handle WebSocket connection
socket.onopen = () => {
    console.log('WebSocket connection established.');
};

socket.onmessage = (event) => {
    // Update prediction from the backend
    const data = JSON.parse(event.data);
    if (data.prediction) {
        predictionText.textContent = data.prediction;
    }
};

socket.onerror = (error) => {
    console.error('WebSocket error:', error);
};

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        webcam.srcObject = stream;
        webcam.onloadedmetadata = () => {
            webcam.play();
            outputCanvas.width = webcam.videoWidth;
            outputCanvas.height = webcam.videoHeight;
            requestAnimationFrame(sendFrameToBackend);
        };
    })
    .catch((error) => {
        console.error('Error accessing webcam:', error);
    });

// Function to send frames to the backend
function sendFrameToBackend() {
    // Draw the current video frame onto the canvas
    ctx.drawImage(webcam, 0, 0, outputCanvas.width, outputCanvas.height);

    // Convert the canvas image to base64
    const frameData = outputCanvas.toDataURL('image/jpeg', 0.8);

    // Send the frame data to the backend via WebSocket
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ frame: frameData }));
    }

    // Request the next frame
    requestAnimationFrame(sendFrameToBackend);
}