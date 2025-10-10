import cv2
import numpy as np
from touch import Touch, jump_gest_detector
import mediapipe
from dino import dino_game
from space_shooter import space_shooter_game
from gesture_space import detect_gestures
from TicTacToe import TicTacToeMain
import subprocess

canvas_width, canvas_height, canvas_scale = 1000, 1000, 9

canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
height, width, _ = canvas.shape
height = int(height*6/10) #virtual height

overlay_coordinates = [
    # --- First Row ---
    [(width * 1 // 3 - width // 8, height * 1 // 3 - height // 8),(width * 1 // 3 + width // 8, height * 1 // 3 + height // 8)],
    [(width * 2 // 3 - width // 8, height * 1 // 3 - height // 8),(width * 2 // 3 + width // 8, height * 1 // 3 + height // 8)],

    # --- Second Row ---
    [(width * 1 // 3 - width // 8, height * 2 // 3 - height // 8),(width * 1 // 3 + width // 8, height * 2 // 3 + height // 8)],
    [(width * 2 // 3 - width // 8, height * 2 // 3 - height // 8),(width * 2 // 3 + width // 8, height * 2 // 3 + height // 8)],
]

icon_path = ["icon_files/icon0.png", "icon_files/icon1.png", "icon_files/icon2.png", "icon_files/icon3.png"]

for i,overlay in enumerate(overlay_coordinates):
    start, end = overlay
    icon = cv2.imread(icon_path[i])
    if icon is not None:
        icon = cv2.resize(icon, (end[0] - start[0], end[1] - start[1]))
        icon = cv2.flip(icon, 0)
        canvas[start[1]:end[1], start[0]:end[0]] = icon
        cv2.rectangle(canvas, (start[0],start[1]), (end[0],end[1]), (0,0,1), 10)


src_pts = np.array([
    [0, 0],
    [canvas_width, 0],
    [canvas_width, canvas_height],
    [0, canvas_height]
], dtype=np.float32)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_height, frame_width, frame_area = None, None, None
warped_canvas = None

H = None # my homeography matrix

pred_count = 0
game_chosen = False

hands_method = mediapipe.solutions.hands.Hands(max_num_hands=1)
space_hands_method = mediapipe.solutions.hands.Hands(max_num_hands=1)
game = dino_game()
game2=space_shooter_game()
next(game)
next(game2) 
key = -1
option=  -1

filtered_pts = None
delay_counter = 30

indexPos = (0,0)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if not frame_height:
        frame_height, frame_width = frame.shape
        frame_area = frame_height*frame_width

    if option != -1:
        sq_width = int(frame_width//1.7)
        x0 = (frame_width - sq_width)//2
        y0 = canvas_width//4

    else:
        sq_width = int(frame_width//0.9)
        x0 = (frame_width - sq_width)//2
        y0 = int(canvas_width//1.02)

    filtered_pts = np.array([
            [sq_width + x0, sq_width - y0],
            [0 + x0, sq_width - y0],
            [0 + x0, 0 - y0],
            [sq_width + x0, 0 - y0],
            
        ], dtype=np.float32)

    
    H, _ = cv2.findHomography(src_pts, filtered_pts)
    
    if H is not None:
        frame2_cpy = frame2.copy()
        if option == -1:
            canvas2, option2, indexPos = Touch(frame2, H, overlay_coordinates, canvas.copy(), hands_method)
            pinch_gesture_detected = jump_gest_detector(frame2, hands_method) #in my case the jump gesture is a pinch gesture
            if delay_counter < 60:
                delay_counter += 1
            else:
                if pinch_gesture_detected:
                    option = option2
        elif option==0:
            delay_counter = 0
            cv2.destroyAllWindows()
            TicTacToeMain(cap)
            option = -1
        elif option==1 :
            delay_counter = 0
            x = frame.copy()
            x = np.fliplr(x)
            gesture_move, gesture_fire, palm_x = detect_gestures(x, hands_method, 1000)
            canvas2= game2.send((gesture_move, gesture_fire, palm_x))
        elif option==2:
            delay_counter = 0
            jump_gesture_detected = jump_gest_detector(frame2, hands_method)
            try:
                if jump_gesture_detected:
                    canvas2 = game.send(True)
                    print("space hit")
                else:
                    canvas2 = game.send(False)
            except StopIteration:
                print("Game generator ended.")
                option = -1

        elif option == 3:
            delay_counter = 0
            print("Releasing camera and running command...")
            command_to_run = ["python", "objectDetection.py"]
            cap.release()
            cv2.destroyAllWindows() 
            subprocess.run(command_to_run)
            print("Re-initializing camera for AR Interface...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("FATAL: Could not re-open camera. Exiting.")
                break
            print("Command finished. Returning to menu.")
            option = -1
            

        # warped perspective... i like to think of it as 'h matrix' takin the src and dst points,
        # making a relation between them and when passed onto wraped perspective, allows
        # any canvas size to be input. the fn simply computes h -1 and copies the pixles
        # wherever it needs to... no stretching etc... cool...

        #  canvas2[:,:,:] = 255

        warped_canvas = cv2.warpPerspective(canvas2, H, (frame_width, frame_height))
        mask = np.any(warped_canvas != 0, axis=2)

        frame2[:, :, :][mask] = warped_canvas[:, :, :][mask]

        cv2.addWeighted(frame2, 0.95, frame2_cpy, 0.05, 0, frame2)
        H = None
    if option == -1:
        cv2.circle(frame2,indexPos,25,(0,0,255),-1)    
    frame2 = np.fliplr(frame2)
    cv2.imshow("video", frame2)
    key = cv2.waitKey(33)
    if key & 0xFF == ord('q'):
        break