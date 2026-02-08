from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import random
import atexit

app = Flask(__name__)

try:
    model = tf.keras.models.load_model("hand_landmark_model.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

CLASS_NAMES = ["Paper", "Rock", "Scissors"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

OPTIONS = ["Rock", "Paper", "Scissors"]

cap = cv2.VideoCapture(0)

user_score = 0
computer_score = 0
game_running = True
last_computer_choice = None

def release_camera():
    cap.release()
    cv2.destroyAllWindows()

atexit.register(release_camera)

def detect_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]

            if model and scaler:
                try:
                    input_data = np.array(landmarks).reshape(1, -1)
                    input_data = scaler.transform(input_data)
                    prediction = model.predict(input_data)
                    return CLASS_NAMES[np.argmax(prediction)]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    return None
            else:
                print("Model or scaler not loaded")
                return None
    return None

def get_unique_computer_choice():
    global last_computer_choice
    new_choice = random.choice(OPTIONS)
    while new_choice == last_computer_choice:
        new_choice = random.choice(OPTIONS)
    last_computer_choice = new_choice
    return new_choice

def determine_winner(user, computer):
    if user == computer:
        return "Tie"
    return "User" if (user == "Rock" and computer == "Scissors") or \
                    (user == "Scissors" and computer == "Paper") or \
                    (user == "Paper" and computer == "Rock") else "Computer"

latest_user_choice = None

def generate_frames():
    global latest_user_choice

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        hand_choice = detect_hand(frame)
        if hand_choice:
            latest_user_choice = hand_choice
            winner_text = f"User: {hand_choice}"    
        else:
            latest_user_choice = None
            winner_text = "No hand detected"

        cv2.putText(frame, winner_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play', methods=['GET'])
def play():
    global user_score, computer_score, game_running, latest_user_choice

    if not game_running:
        game_running = True

    user_choice = latest_user_choice

    if user_choice is None:
        return jsonify({"winner": "No hand detected", "user": "None", "computer": None})

    computer_choice = get_unique_computer_choice()
    winner = determine_winner(user_choice, computer_choice)

    if winner == "User":
        user_score += 1
    elif winner == "Computer":
        computer_score += 1

    return jsonify({
        "winner": winner,
        "user": user_choice,
        "computer": computer_choice,
        "user_score": user_score,
        "computer_score": computer_score
    })

@app.route('/reset', methods=['POST'])
def reset_scores():
    global user_score, computer_score
    user_score, computer_score = 0, 0
    return jsonify({"message": "Scores reset", "user_score": 0, "computer_score": 0})

@app.route('/stop_game', methods=['GET'])
def stop_game():
    final_winner = "Tie" if user_score == computer_score else ("User" if user_score > computer_score else "Computer")
    return jsonify({"final_winner": final_winner})

if __name__ == '__main__':
    app.run(debug=True)