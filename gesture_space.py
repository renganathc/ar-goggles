import cv2
import numpy as np
import mediapipe as mp

WIDTH, HEIGHT = 1000, 700
# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand landmark indices
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

def is_palm(lm):
    
    # Detecs if the hand is shown an open palm.
    #Returns True if the palm is facing camera (all fingers above wrist).
    
    return (lm[INDEX_TIP].y < lm[WRIST].y and
            lm[MIDDLE_TIP].y < lm[WRIST].y and
            lm[RING_TIP].y < lm[WRIST].y and
            lm[PINKY_TIP].y < lm[WRIST].y)

def is_index_up(lm):
    
    #Returns True if index finger tip is above other finger tips if so it starts firing 
    
    return (lm[INDEX_TIP].y < lm[MIDDLE_TIP].y and
            lm[INDEX_TIP].y < lm[RING_TIP].y and
            lm[INDEX_TIP].y < lm[PINKY_TIP].y)

def detect_gestures(frame, hands_method, frame_width, frame_height):
    
    gesture_move = False
    gesture_fire = False
    palm_x = frame_width // 2  # default player x if no hand

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_method.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        if is_palm(lm):
            palm_x = int(lm[WRIST].x * frame_width)
            gesture_move = True

        if is_index_up(lm):
            gesture_fire = True

    return gesture_move, gesture_fire, palm_x
