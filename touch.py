import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#hands_method = mphands.Hands(max_num_hands=1)

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def Touch(frame, element_coordinates, hands_method):
    index_pos = (-1, -1)
    element = -1  # Default: no icon selected

    # Convert frame for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_method.process(frame_rgb)

    frame_height, frame_width = frame.shape[:2]

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Pixel position of index finger
        index_pos = (int(index.x * frame_width), int(index.y * frame_height))

        # Check which icon region the finger is in
        for i, ((x0, y0), (x1, y1)) in enumerate(element_coordinates):
            if x0 < index_pos[0] < x1 and y0 < index_pos[1] < y1:
                element = i
                break  # Stop once we find the active element

    return element, index_pos

def jump_gest_detector(frame, hands_method):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    result = hands_method.process(frame)
    h, w = frame.shape[:2]
    jump_gesture_detected = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
        index_base, pinky_mcp = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], hand.landmark[mp_hands.HandLandmark.PINKY_MCP]

        # Get pixel coords
        ix, iy = int(index.x * w), int(index.y * h)
        tx, ty = int(thumb.x * w), int(thumb.y * h)
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        mx, my = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

        # Hand size = distance between wrist and middle fingertip
        hand_size = ((mx - wx)**2 + (my - wy)**2)**0.5
        pinch_dist = ((ix - tx)**2 + (iy - ty)**2)**0.5

        # Detect pinch when fingers are very close compared to hand size
        if pinch_dist < 0.4 * hand_size:
            jump_gesture_detected = True

    return jump_gesture_detected

def jump_gest_detector2(frame, hands_method):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
	result = hands_method.process(frame)
	h, w = frame.shape[:2]
	jump_gesture_detected = False

	if result.multi_hand_landmarks:
		hand_landmarks = result.multi_hand_landmarks[0]
		index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
		thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

		ix, iy = int(index.x * w), int(index.y * h)
		tx, ty = int(thumb.x * w), int(thumb.y * h)

		distance = ((ix - tx)**2 + (iy - ty)**2)**0.5
		if distance < int(h/11.5): # sensitivity parameter :)
			jump_gesture_detected = True

	return jump_gesture_detected