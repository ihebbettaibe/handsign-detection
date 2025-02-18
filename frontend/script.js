const elements = {
    webcam: document.getElementById('webcam'),
    outputCanvas: document.getElementById('outputCanvas'),
    predictionText: document.querySelector('#predicted-sign .value'),
    currentWordText: document.querySelector('#current-word .value'),
    phraseText: document.querySelector('#phrase .value'),
    savedPhrasesDiv: document.getElementById('saved-phrases'),
    clearButton: document.getElementById('clearPhrases'),
    websocket: socket // if needed elsewhere
};

const socket = new WebSocket("ws://127.0.0.1:8000/ws");

let currentWord = '';
let phrase = [];
let savedPhrases = [];
let lastSign = '';
let noSignTimeout = null;

// Update detected items with some basic visual feedback
function updateDetectionBox(element, text) {
    element.textContent = text;
    element.parentElement.classList.add('active');
    setTimeout(() => {
        element.parentElement.classList.remove('active');
    }, 300);
}

// WebSocket message handling (assuming socket initialization is done)
socket.onmessage = (event) => {
    console.log("WebSocket message received:", event.data); // Debug log
    try {
        const data = JSON.parse(event.data);
        if (data.prediction) {
            const sign = data.prediction;
            console.log("Predicted", sign); // Debug log

            // Update the UI for the predicted sign
            elements.predictionText.textContent = sign;

            if (sign !== lastSign && sign !== 'No hands detected') {
                currentWord += sign;
                elements.currentWordText.textContent = currentWord;
                lastSign = sign;

                clearTimeout(noSignTimeout);
                noSignTimeout = setTimeout(finalizeWord, 4000);
            }
        }
    } catch (error) {
        console.error('Error parsing WebSocket message:', error);
    }
};

function finalizeWord() {
    if (currentWord) {
        phrase.push(currentWord);
        if (phrase.length > 4) {
            savedPhrases.push([...phrase]);
            phrase = [];
            updateSavedPhrases();
        }
        elements.phraseText.textContent = phrase.join(' ');
        currentWord = '';
        elements.currentWordText.textContent = '';
        lastSign = '';
    }
}

function updateSavedPhrases() {
    elements.savedPhrasesDiv.innerHTML = savedPhrases
        .map(phrase => `<div class="saved-phrase">${phrase.join(' ')}</div>`)
        .join('');
}

elements.clearButton.addEventListener('click', () => {
    savedPhrases = [];
    updateSavedPhrases();
});

const webcam = document.getElementById('webcam');

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            webcam.srcObject = stream;
            webcam.play();
        })
        .catch(err => {
            console.error("Error accessing camera: ", err);
        });
} else {
    alert("getUserMedia not supported in this browser.");
}