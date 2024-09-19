# finger size

import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_finger = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            distance = math.sqrt((index_finger.x - thumb_finger.x) ** 2 + (index_finger.y - thumb_finger.y) ** 2)

            max_distance = 0.2 
            volume = int(distance / max_distance * 100)
            print("Volume:", volume)  

    cv2.imshow("Image", img)
    try:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        exit()
