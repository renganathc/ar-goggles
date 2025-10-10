import cv2
import numpy as np
from touch import Touch, jump_gest_detector
import mediapipe
from dino import dino_game
from space_shooter import space_shooter_game
from gesture_space import detect_gestures
from TicTacToe import TicTacToeMain
import subprocess

# --- Config ---
canvas_width, canvas_height = 1920, 1080
height, width = canvas_height, canvas_width

overlay_coordinates = [
    # First Row
    [(width * 1 // 4 - width // 6, height * 1 // 4 - height // 6),
     (width * 1 // 4 + width // 6, height * 1 // 4 + height // 6)],

    [(width * 3 // 4 - width // 6, height * 1 // 4 - height // 6),
     (width * 3 // 4 + width // 6, height * 1 // 4 + height // 6)],

    # Second Row
    [(width * 1 // 4 - width // 6, height * 3 // 4 - height // 6),
     (width * 1 // 4 + width // 6, height * 3 // 4 + height // 6)],

    [(width * 3 // 4 - width // 6, height * 3 // 4 - height // 6),
     (width * 3 // 4 + width // 6, height * 3 // 4 + height // 6)],
]

# --- Load icons ---
icons = []
for path in ["icon_files/icon0.png", "icon_files/icon1.png",
             "icon_files/icon2.png", "icon_files/icon3.png"]:
    icon = cv2.imread(path)  # BGR, fully opaque
    if icon is None:
        print(f"Failed to load {path}")
        icon = np.zeros((100, 100, 3), dtype=np.uint8)
    icons.append(icon)

# --- Camera & Mediapipe ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
hands_method = mediapipe.solutions.hands.Hands(max_num_hands=1)
space_hands_method = mediapipe.solutions.hands.Hands(max_num_hands=1)

# --- Game generators ---
game = dino_game()
game2 = space_shooter_game()
next(game)
next(game2)

option = -1
delay_counter = 0
indexPos = (0, 0)

# --- Hover selection variables ---
hover_start_time = None
hovered_element = -1
HOVER_DURATION = 2.0  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    display = frame.copy()

    # --- Overlay icons ---
    for i, ((x0, y0), (x1, y1)) in enumerate(overlay_coordinates):
        icon_resized = cv2.resize(icons[i], (x1 - x0, y1 - y0))
        display[y0:y1, x0:x1] = icon_resized

    # --- Menu logic ---
    # --- Menu logic ---
    if option == -1:
        element, indexPos = Touch(frame, overlay_coordinates, hands_method)

        for i, ((x0, y0), (x1, y1)) in enumerate(overlay_coordinates):
            # Draw black border for all boxes
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 0, 0), 20)

        if element != -1:
            (x0, y0), (x1, y1) = overlay_coordinates[element]

            # Compute hover progress fraction
            if hover_start_time is not None:
                elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - hover_start_time
                progress = min(elapsed / HOVER_DURATION, 1.0)
            else:
                progress = 0

            # Draw proportional red border
            border_length_x = int((x1 - x0) * progress)
            border_length_y = int((y1 - y0) * progress)

            # Top border
            cv2.line(display, (x0, y0), (x0 + border_length_x, y0), (0, 0, 255), 20)
            # Right border
            cv2.line(display, (x1, y0), (x1, y0 + border_length_y), (0, 0, 255), 20)
            # Bottom border
            cv2.line(display, (x1, y1), (x1 - border_length_x, y1), (0, 0, 255), 20)
            # Left border
            cv2.line(display, (x0, y1), (x0, y1 - border_length_y), (0, 0, 255), 20)

            element, indexPos = Touch(frame, overlay_coordinates, hands_method)

        # --- Hover timer logic ---
        if element == hovered_element:
            if hover_start_time is None:
                hover_start_time = cv2.getTickCount() / cv2.getTickFrequency()
            else:
                elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - hover_start_time
                if elapsed >= HOVER_DURATION:
                    option = element  # select
                    hover_start_time = None
                    hovered_element = -1
        else:
            hovered_element = element
            hover_start_time = None

        if delay_counter < 20:
            delay_counter += 1
            if delay_counter == 1:
                game = dino_game()
                next(game)
                game2 = space_shooter_game()
                next(game2)

        if indexPos != (-1, -1):
            cv2.circle(display, indexPos, 35, (0, 0, 255), -1)

    # --- Selected options ---
    elif option == 0:
        delay_counter = 0
        cv2.destroyAllWindows()
        TicTacToeMain(cap)
        option = -1

    elif option == 1:
        delay_counter = 0
        x = np.fliplr(frame.copy())
        gesture_move, gesture_fire, palm_x = detect_gestures(x, hands_method, 1000)
        display = game2.send((gesture_move, gesture_fire, palm_x))

    elif option == 2:
        delay_counter = 0
        jump_gesture_detected = jump_gest_detector(frame, hands_method)
        try:
            display = game.send(jump_gesture_detected)
        except StopIteration:
            print("Game ended.")
            option = -1

    elif option == 3:
        delay_counter = 0
        print("Running object detection...")
        cap.release()
        cv2.destroyAllWindows()
        subprocess.run(["python", "objectDetection.py"])
        cap = cv2.VideoCapture(0)
        option = -1

    cv2.imshow("Canvas Display", display)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('r'):
        option = -1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
