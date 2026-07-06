import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key

keyboard = Controller()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0

gesture_delay = 20
counter = 0

cap = cv2.VideoCapture(0)

def fingers_up(hand):
    tips = [8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand.landmark[4].x < hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in tips:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if counter > 0:
        counter -= 1

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            fingers = fingers_up(handLms)

            x = int(handLms.landmark[8].x * frame.shape[1])
            y = int(handLms.landmark[8].y * frame.shape[0])

            # ✏️ DRAW (index finger)
            if fingers[1] == 1 and fingers[2] == 0:
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 255), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            # 👉 POINTER (2 fingers)
            if fingers[1] == 1 and fingers[2] == 1:
                cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

            # ➡️ NEXT SLIDE
            if fingers == [0,1,0,0,0] and x > frame.shape[1] - 100 and counter == 0:
                keyboard.press(Key.right)
                keyboard.release(Key.right)
                counter = gesture_delay

            # ⬅️ PREVIOUS SLIDE
            if fingers == [0,1,0,0,0] and x < 100 and counter == 0:
                keyboard.press(Key.left)
                keyboard.release(Key.left)
                counter = gesture_delay

            # 🧹 CLEAR (fist)
            if fingers == [0,0,0,0,0]:
                canvas = np.zeros_like(frame)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # ✅ IMPORTANT FIX → drawing stays permanently
    frame = cv2.add(frame, canvas)

    cv2.imshow("Touchless Presentation Tool", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()