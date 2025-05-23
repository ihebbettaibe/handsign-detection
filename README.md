# 👐 HandSign Detection

> Real-time sign language detection and interpretation using machine learning

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

<p align="center">
  <img src="https://raw.githubusercontent.com/ihebbettaibe/handsign-detection/main/docs/demo.gif" alt="HandSign Detection Demo" />
</p>

## 🌟 Overview

**HandSign Detection** is an innovative project that uses computer vision and machine learning techniques to recognize and interpret sign language gestures in real-time. By leveraging the power of TensorFlow and OpenCV, this application bridges communication gaps for the deaf and hard-of-hearing community.

The system captures hand movements through a webcam, processes the visual data through a trained neural network, and translates the detected gestures into text and speech, making sign language accessible to everyone.

## 💡 Key Features

- 👋 **Real-time Hand Gesture Recognition** with high accuracy
- 🔠 **Multi-class Classification** of American Sign Language (ASL) alphabet
- 🎥 **Live Video Processing** using OpenCV
- 🧠 **Custom-trained Neural Network** optimized for hand pose detection
- 🔊 **Text-to-Speech Conversion** for detected signs (optional)
- 📊 **Confidence Metrics** for detection reliability
- 📱 **User-friendly Interface** with minimal setup required

## 🛠️ Tech Stack

- **Python 3.7+**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model training and inference
- **OpenCV**: Computer vision library for image processing and webcam integration
- **MediaPipe**: Hand landmark detection and tracking
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization of training results and model performance

## 🔌 System Architecture

```
[Webcam Input] → [Hand Detection & Tracking] → [Feature Extraction] → [ML Classification] → [Sign Interpretation] → [User Interface]
```

- Webcam captures continuous video feed
- MediaPipe isolates and tracks hand landmarks
- Features are extracted and normalized for the model
- TensorFlow model classifies the hand gesture
- Results are displayed with confidence scores
- Detected sign is converted to text/speech (optional)

## 📊 Model Performance

- **Accuracy**: ~95% on test dataset
- **Classes**: 26 ASL alphabet signs + custom gestures
- **Training Data**: 2000+ labeled hand gesture images
- **Validation Method**: 5-fold cross-validation

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam or camera device
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ihebbettaibe/handsign-detection.git
cd handsign-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the real-time detection:
```bash
python src/main.py
```

2. Position your hand in front of the camera
3. Make ASL signs and see the real-time detection results

### Training Your Own Model

1. Prepare your dataset in the `data/` directory
2. Run the training script:
```bash
python src/train.py --epochs 50 --batch-size 32
```

3. Evaluate the model:
```bash
python src/evaluate.py --model models/handsign_model.h5
```

## 📂 Project Structure

```
handsign-detection/
├── data/                  # Training and testing datasets
│   ├── raw/               # Raw collected images
│   └── processed/         # Preprocessed images
├── models/                # Trained model files
│   └── handsign_model.h5  # Main model
├── src/                   # Source code
│   ├── main.py            # Application entry point
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation script
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── model.py           # Model architecture definition
│   └── utils/             # Utility functions
├── docs/                  # Documentation
│   └── demo.gif           # Demo animation
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🔄 Future Enhancements

- [ ] Expand to dynamic gesture recognition (moving signs)
- [ ] Mobile application development
- [ ] Support for multiple sign language systems (BSL, JSL, etc.)
- [ ] Improved accessibility features
- [ ] Edge device optimization for offline use
- [ ] Web API for integration with other applications

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Sign language datasets contributors
- TensorFlow and OpenCV communities
- MediaPipe team for their hand tracking solution
- All contributors who have helped shape this project

## 📬 Contact

<p align="center">
  <a href="mailto:iheb.bentaieb@supcom.tn"><img src="https://img.shields.io/badge/Email-iheb.bentaieb%40supcom.tn-blue?style=for-the-badge&logo=microsoft-outlook"></a>
  <a href="https://github.com/ihebbettaibe"><img src="https://img.shields.io/badge/GitHub-ihebbettaibe-black?style=for-the-badge&logo=github"></a>
</p>

---

<p align="center">Making communication accessible for everyone through technology</p>
