const webcam = document.getElementById('webcam');
const outputCanvas = document.getElementById('outputCanvas');
const predictionText = document.getElementById('predicted-sign');
const ctx = outputCanvas.getContext('2d');

// Initialize WebSocket
const socket = new WebSocket('ws://127.0.0.1:8000/ws'); // Ensure the port matches the FastAPI server

// Handle WebSocket events
socket.onopen = () => {
    console.log('WebSocket connection established.');
};

socket.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        if (data.prediction) {
            predictionText.textContent = `Predicted Sign: ${data.prediction}`;
        }
    } catch (error) {
        console.error('Error parsing WebSocket message:', error);
    }
};

socket.onerror = (error) => {
    console.error('WebSocket error:', error);
};

socket.onclose = () => {
    console.log('WebSocket connection closed.');
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

    // Remove the prefix from the base64 string
    const base64Data = frameData.split(',')[1];

    // Send the frame data to the backend via WebSocket
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ frame: base64Data }));
    }

    // Request the next frame (limit frame rate to 10 FPS)
    setTimeout(() => {
        requestAnimationFrame(sendFrameToBackend);
    }, 100); // Adjust the delay to control the frame rate
}

// Close WebSocket connection when the page is unloaded
window.addEventListener('beforeunload', () => {
    if (socket.readyState === WebSocket.OPEN) {
        socket.close();
    }
});
