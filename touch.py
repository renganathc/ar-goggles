import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#hands_method = mphands.Hands(max_num_hands=1)

def Touch(frame, H, element_coordinates, canvas, hands_method):
	index_pos = (-1,-1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
	result = hands_method.process(frame)
	frame_height, frame_width = frame.shape[:2]
	element = -1

	if result.multi_hand_landmarks:
		hand_landmarks = result.multi_hand_landmarks[0]
		index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
		thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

		index_pos = int(index.x * frame_width),int(index.y * frame_height)
		H_inv = np.linalg.inv(H)

		FingerPosOnUI = cv2.perspectiveTransform(np.array([[index_pos]], dtype=np.float32), H_inv)
		fx,fy = int(FingerPosOnUI[0][0][0]),int(FingerPosOnUI[0][0][1])
		for i,((x0,y0),(x1,y1)) in enumerate(element_coordinates):
				if x0<fx<x1 and y0<fy<y1:
					element = i
					cv2.rectangle(canvas, (x0,y0), (x1,y1), (0,0,255), 10)
					break

	return canvas,element,index_pos

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
        index_base = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

        # Get pixel coords
        ix, iy = int(index.x * w), int(index.y * h)
        tx, ty = int(thumb.x * w), int(thumb.y * h)
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        mx, my = int(index_base.x * w), int(index_base.y * h)

        # Hand size = distance between wrist and middle fingertip
        hand_size = ((mx - wx)**2 + (my - wy)**2)**0.5
        pinch_dist = ((ix - tx)**2 + (iy - ty)**2)**0.5

        # Detect pinch when fingers are very close compared to hand size
        if pinch_dist < 0.35 * hand_size:
            jump_gesture_detected = True

    return jump_gesture_detected
