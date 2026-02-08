# 🤖 BeatBot – Rock-Paper-Scissors AI

BeatBot is an interactive Rock-Paper-Scissors game that leverages **Computer Vision** to detect hand gestures from a live webcam feed. Challenge a smart AI opponent in real-time through either a local Python application or a sleek Flask-based web interface.

---

## 🌟 Key Features

* **Real-time Gesture Recognition:** Powered by Mediapipe for high-accuracy hand landmark detection.
* **Dual-Mode Interface:** Play via a standalone OpenCV window or a modern Web browser.
* **Smart AI Opponent:** Uses logic to ensure fair play (prevents predictable move repetition).
* **Interactive Scoreboard:** Dynamic round tracking and win/loss feedback.
* **Gesture-Driven Utilities:** Optional feature to control screen brightness using thumb-index finger distance.
* **Full ML Pipeline:** Includes scripts for data collection, dataset creation, and model training.

---

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Computer Vision:** OpenCV, Mediapipe
* **Machine Learning:** TensorFlow / Keras, NumPy, Scikit-learn
* **Web Framework:** Flask
* **Frontend:** HTML, CSS, JavaScript
* **System Tools:** screen_brightness_control (Optional)

---

## 📁 Project Structure

```text
BeatBot/
├── image_data/                # Raw hand gesture images (Paper/Rock/Scissors)
├── processed_images/          # Images with drawn landmarks for training
├── images/                    # Reference images for AI moves
├── static/                    # Static assets (CSS, JS, Web images)
├── templates/                 # HTML templates (game.html, test.html)
├── collect_imgs.py            # Script to capture hand images via webcam
├── create_dataset.py          # Converts images to dataset CSV
├── hand_gesture_detetc.py     # Standalone hand gesture detection app
├── train_model.py             # Script to train gesture classification model
├── hand_landmarks_dataset.csv # Generated dataset from images
├── hand_landmark_model.h5     # Trained Keras classification model
├── scaler.pkl                 # StandardScaler object for normalization
├── flaskApp.py                # Flask backend with live webcam RPS game
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```
---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/alisahito17/BeatBot-RockPaperScissor-Using-Computer-Vision
cd BeatBot
pip install -r requirements.txt
```

### 2. Running the Game

#### Local Python Application
Run the standalone OpenCV detection:
python hand_gesture_detetc.py

#### Flask Web Interface
1. Start the Flask server: python flaskApp.py
2. Open your browser and go to: http://127.0.0.1:5000/game

---

## 🧠 Model & AI Logic

### Training Process
1. Data Collection: collect_imgs.py captures gestures and stores them in image_data/.
2. Feature Extraction: create_dataset.py extracts 42 specific features (21 hand landmarks × x, y coordinates).
3. Classification: A Keras model classifies the landmarks into three classes: Rock, Paper, or Scissors.
4. Normalization: scaler.pkl ensures real-time input coordinates are scaled correctly.

### AI Strategy
The AI move is randomized but includes a logic check to avoid repeating its previous move, making the gameplay more engaging and fair.

---

## 👤 Author
**alisahito17** – AI Developer

---
*Developed to combine Computer Vision, ML, and Web technologies.*
