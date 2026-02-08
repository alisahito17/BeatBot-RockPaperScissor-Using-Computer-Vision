import os
import pickle
import mediapipe as mp
import cv2
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './image_data'
SAVE_DIR = './processed_images'

os.makedirs(SAVE_DIR, exist_ok=True)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(SAVE_DIR, dir_)
    os.makedirs(class_dir, exist_ok=True)

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img_path_full = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(img_path_full)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

            save_path = os.path.join(class_dir, img_path)
            cv2.imwrite(save_path, img)

hands.close()

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('hand_landmarks_dataset.csv', index=False)

print(f"Dataset saved as 'hand_landmarks_dataset.csv'.")
print(f"Processed images saved in '{SAVE_DIR}/'.")