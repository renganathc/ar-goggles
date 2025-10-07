import cv2
import numpy as np
import pupil_apriltags as apriltag
from touch import Touch, jump_gest_detector
import mediapipe
from dino import dino_game

def create_kf():
    kf = cv2.KalmanFilter(4,2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)

    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

    return kf

def scale_canvas(pts, scale, overlay=0.4):
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    
    bottom_mid = (pts[2] + pts[3]) / 2.0
    dx, dy = bottom_mid - np.array([cx, cy])
    cx += overlay * dx
    cy += overlay * dy

    enlarged_pts = []
    for (x, y) in pts:
        new_x = cx + (x - cx) * scale
        new_y = cy + (y - cy) * scale
        enlarged_pts.append([new_x, new_y])
    pts = np.array(enlarged_pts, dtype=np.float32)
    return pts

detector = apriltag.Detector(
                            families="tag16h5",
                            #quad_sigma=1,      # helps detect blurry edges
                            #refine_edges=True,
                            #decode_sharpening=0.5
                            )

canvas_width, canvas_height, canvas_scale = 1000, 1000, 7

canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
canvas[:,:,2] = 100
height, width, _ = canvas.shape
height = int(height*6/10) #virtual height

overlay_coordinates = [
    [(width//4 - width//11, height//2 - height//3),(width//4 + width//11, height//2 + height//3)],
    [(width*2//4 - width//11, height//2 - height//3),(width*2//4 + width//11, height//2 + height//3)],
    [(width*3//4 - width//11, height//2 - height//3),(width*3//4 + width//11, height//2 + height//3)],
    ]

for i,overlay in enumerate(overlay_coordinates):
    start, end = overlay

    if i == 0:
        icon = cv2.imread("icon_files/tic_icon.png")
    elif i == 1:
        icon = cv2.imread("icon_files/connect_icon.png")
    elif i == 2:
        icon = cv2.imread("icon_files/yolo_icon.png")

    icon = cv2.resize(icon, (end[0] - start[0], end[1] - start[1]))
    canvas[start[1]:end[1], start[0]:end[0]] = icon[::-1, :]

src_pts = np.array([
    [0, 0],
    [canvas_width, 0],
    [canvas_width, canvas_height],
    [0, canvas_height]
], dtype=np.float32)

cap = cv2.VideoCapture(0)

frame_height, frame_width, frame_area = None, None, None
warped_canvas = None

kf0, kf1, kf2, kf3 = create_kf(), create_kf(), create_kf(), create_kf()
kf_init = False
H = None # my homeography matrix

pred_count = 0
game_chosen = False

hands_method = mediapipe.solutions.hands.Hands(max_num_hands=1)

game = dino_game()
next(game)

key = -1

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if not frame_height:
        frame_height, frame_width = frame.shape
        frame_area = frame_height*frame_width

    results = detector.detect(frame)
    marker_found = False

    if results:
        for r in results:
            area = cv2.contourArea(np.array(r.corners, dtype=np.int32))
            if r.tag_id == 0 and area > frame_area*0.002: #adjust min area
                dst_pts = np.array(r.corners, dtype=np.float32)
                #print(dst_pts)
                if dst_pts.shape != (4,2):
                    continue

                if not kf_init:
                    kf0.statePre, kf0.statePost  = np.array([[dst_pts[0][0]], [dst_pts[0][1]], [0], [0]], dtype=np.float32), kf0.statePre.copy()
                    kf1.statePre, kf1.statePost  = np.array([[dst_pts[1][0]], [dst_pts[1][1]], [0], [0]], dtype=np.float32), kf1.statePre.copy()
                    kf2.statePre, kf2.statePost  = np.array([[dst_pts[2][0]], [dst_pts[2][1]], [0], [0]], dtype=np.float32), kf2.statePre.copy()
                    kf3.statePre, kf3.statePost  = np.array([[dst_pts[3][0]], [dst_pts[3][1]], [0], [0]], dtype=np.float32), kf3.statePre.copy()
                    kf_init = True

                else:
                    kf0.predict()
                    kf0.correct(np.array([[dst_pts[0][0]], [dst_pts[0][1]]], dtype=np.float32))
                    kf1.predict()
                    kf1.correct(np.array([[dst_pts[1][0]], [dst_pts[1][1]]], dtype=np.float32))
                    kf2.predict()
                    kf2.correct(np.array([[dst_pts[2][0]], [dst_pts[2][1]]], dtype=np.float32))
                    kf3.predict()
                    kf3.correct(np.array([[dst_pts[3][0]], [dst_pts[3][1]]], dtype=np.float32))

                    # filtered_pts = np.zeros((4, 2), dtype=np.float32)
                    # filtered_pts[0] = kf0.statePost[:2].flatten()
                    # filtered_pts[1] = kf1.statePost[:2].flatten()
                    # filtered_pts[2] = kf2.statePost[:2].flatten()
                    # filtered_pts[3] = kf3.statePost[:2].flatten()

                    dst_pts = scale_canvas(dst_pts, canvas_scale)
                    H, _ = cv2.findHomography(src_pts, dst_pts)
                    
                pred_count = 0
                marker_found = True
                break

    if kf_init and pred_count <= 3 and not marker_found:
        pred_count += 1
        filtered_pts = np.zeros((4, 2), dtype=np.float32)

        if pred_count <= 2:
            filtered_pts[0] = kf0.predict()[:2].flatten()
            filtered_pts[1] = kf1.predict()[:2].flatten()
            filtered_pts[2] = kf2.predict()[:2].flatten()
            filtered_pts[3] = kf3.predict()[:2].flatten()
        
        else:
            filtered_pts[0] = kf0.statePost[:2].flatten()
            filtered_pts[1] = kf1.statePost[:2].flatten()
            filtered_pts[2] = kf2.statePost[:2].flatten()
            filtered_pts[3] = kf3.statePost[:2].flatten()

        filtered_pts = scale_canvas(filtered_pts, canvas_scale)
        H, _ = cv2.findHomography(src_pts, filtered_pts)
    
    if H is not None:
        frame2_cpy = frame2.copy()
        if not game_chosen:
            canvas2, option = Touch(frame2, H, overlay_coordinates, canvas.copy(), hands_method)
            if option == 0 :
                game_chosen = True
        else:
            jump_gesture_detected = jump_gest_detector(frame2, hands_method)
            if jump_gesture_detected:
                canvas2 = game.send(True)
                print("space hit")
            else:
                canvas2 = game.send(False)

        # warped perspective... i like to think of it as 'h matrix' takin the src and dst points,
        # making a relation between them and when passed onto wraped perspective, allows
        # any canvas size to be input. the fn simply computes h -1 and copies the pixles
        # wherever it needs to... no stretching etc... cool...

        warped_canvas = cv2.warpPerspective(canvas2, H, (frame_width, frame_height))
        mask = np.any(warped_canvas != 0, axis=2)

        frame2[:, :, :][mask] = warped_canvas[:, :, :][mask]

        cv2.addWeighted(frame2, 0.9, frame2_cpy, 0.1, 0, frame2)
        H = None

    cv2.imshow("video", frame2)
    key = cv2.waitKey(33)
    if key & 0xFF == ord('q'):
        break