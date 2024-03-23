import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model("mp_hand_gesture")

f = open("gesture.names", "r")
classNames = f.read().split("\n")
f.close()
print(classNames)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
result = hands.process(framergb)

className = ''

if result.multi_hand_landmarks:
    landmarks = []
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            lmx = int(lm.x * x)
            lmy = int(lm.y * y)
            landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

