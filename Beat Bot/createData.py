import os
import cv2
import mediapipe as mp
import csv

IMAGE_FOLDER = "image_data"
OUTPUT_CSV = "hand_landmarks_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    writer.writerow(header)

    for label in os.listdir(IMAGE_FOLDER):
        label_path = os.path.join(IMAGE_FOLDER, label)
        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read {img_path}. Skipping...")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                writer.writerow([label] + landmarks)
                print(f"Processed {img_path}")
            else:
                print(f"No hand landmarks detected in {img_path}. Skipping...")

hands.close()
print(f"Landmark dataset saved to {OUTPUT_CSV}")