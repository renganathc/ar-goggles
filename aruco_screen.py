import cv2
import numpy as np
import pupil_apriltags as apriltag
from touch import Touch
import mediapipe

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

def scale_canvas(pts, scale, overlay=0.7):
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

canvas_width, canvas_height, canvas_scale = 1200, 1200, 9

canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
height, width, _ = canvas.shape

overlay_coordinates = [
    [(width//4 - width//11, height//2 - height//6),(width//4 + width//11, height//2 + height//6)],
    [(width*2//4 - width//11, height//2 - height//6),(width*2//4 + width//11, height//2 + height//6)],
    [(width*3//4 - width//11, height//2 - height//6),(width*3//4 + width//11, height//2 + height//6)],
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
    icon = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
    canvas[start[1]:end[1], start[0]:end[0]] = icon

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

hands_method = mediapipe.solutions.hands.Hands(max_num_hands=1)

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
        canvas2 = Touch(frame2, H, overlay_coordinates, canvas.copy(), hands_method)
        warped_canvas = cv2.warpPerspective(canvas2, H, (frame_width, frame_height))
        warped_canvas = cv2.resize(warped_canvas, (frame_width, frame_height))
        mask = np.any(warped_canvas != 0, axis=2)

        for c in range(3):
            frame2[:, :, c][mask] = warped_canvas[:, :, c][mask]

        cv2.addWeighted(frame2, 0.8, frame2_cpy, 0.2, 0, frame2)
        H = None

    cv2.imshow("video", frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break