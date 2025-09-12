import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

canvas_width, canvas_height = 800, 600

canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
height, width, _ = canvas.shape

overlay_coordinates = [
    [(width//4 - width//9, height//2 - height//3),(width//4 + width//9, height//2 + height//3)],
    [(width*2//4 - width//9, height//2 - height//3),(width*2//4 + width//9, height//2 + height//3)],
    [(width*3//4 - width//9, height//2 - height//3),(width*3//4 + width//9, height//2 + height//3)],
    ]

for overlay in overlay_coordinates:
    start, end = overlay
    cv2.rectangle(canvas, start, end, (0, 0, 255), -1)

src_pts = np.array([
    [0, 0],
    [canvas_width, 0],
    [canvas_width, canvas_height],
    [0, canvas_height]
], dtype=np.float32)

cap = cv2.VideoCapture(0)

frame_height, frame_width = None, None

warped_canvas = None

kf0, kf1, kf2, kf3 = cv2.KalmanFilter(4,2), cv2.KalmanFilter(4,2), cv2.KalmanFilter(4,2), cv2.KalmanFilter(4,2)
kf_init = False

kf0.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)

kf0.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=np.float32)

kf0.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
kf0.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

kf1.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)

kf1.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=np.float32)

kf1.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
kf1.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

kf2.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)

kf2.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=np.float32)

kf2.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
kf2.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

kf3.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)

kf3.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=np.float32)

kf3.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
kf3.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

pred_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not frame_height:
        frame_height, frame_width, _ = frame.shape

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None:
        ids = ids.flatten()
        dst_pts = np.zeros((4, 2), dtype=np.float32)

        print(corners)

        if 0 in ids:

            index = np.where(ids == 0)
            dst_pts[0] = corners[0][0][0]
            dst_pts[1] = corners[0][0][1]
            dst_pts[2] = corners[0][0][2]
            dst_pts[3] = corners[0][0][3]

            if not kf_init:
                kf0.statePre  = np.array([[dst_pts[0][0]], [dst_pts[0][1]], [0], [0]], dtype=np.float32)
                kf0.statePost = kf0.statePre.copy()
                kf1.statePre  = np.array([[dst_pts[1][0]], [dst_pts[1][1]], [0], [0]], dtype=np.float32)
                kf1.statePost = kf1.statePre.copy()
                kf2.statePre  = np.array([[dst_pts[2][0]], [dst_pts[2][1]], [0], [0]], dtype=np.float32)
                kf2.statePost = kf2.statePre.copy()
                kf3.statePre  = np.array([[dst_pts[3][0]], [dst_pts[3][1]], [0], [0]], dtype=np.float32)
                kf3.statePost = kf3.statePre.copy()
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

                filtered_pts = np.zeros((4, 2), dtype=np.float32)

                filtered_pts[0] = kf0.statePost[:2].flatten()
                filtered_pts[1] = kf1.statePost[:2].flatten()
                filtered_pts[2] = kf2.statePost[:2].flatten()
                filtered_pts[3] = kf3.statePost[:2].flatten()

                H, _ = cv2.findHomography(src_pts, filtered_pts)
                warped_canvas = cv2.warpPerspective(canvas, H, (frame_width, frame_height))
                warped_canvas = cv2.resize(warped_canvas, (frame_width, frame_height))
                mask = np.any(warped_canvas != 0, axis=2)  # True where at least one channel is non-black
                
                for c in range(3):  # BGR channels
                    frame[:, :, c][mask] = warped_canvas[:, :, c][mask]
                
            pred_count = 0

    elif kf_init and pred_count <= 5:
        pred_count += 1
        filtered_pts = np.zeros((4, 2), dtype=np.float32)

        if pred_count <=3:
            filtered_pts[0] = kf0.predict()[:2].flatten()
            filtered_pts[1] = kf1.predict()[:2].flatten()
            filtered_pts[2] = kf2.predict()[:2].flatten()
            filtered_pts[3] = kf3.predict()[:2].flatten()
        
        else:
            filtered_pts[0] = kf0.statePost[:2].flatten()
            filtered_pts[1] = kf1.statePost[:2].flatten()
            filtered_pts[2] = kf2.statePost[:2].flatten()
            filtered_pts[3] = kf3.statePost[:2].flatten()

        H, _ = cv2.findHomography(src_pts, filtered_pts)
        warped_canvas = cv2.warpPerspective(canvas, H, (frame_width, frame_height))
        mask = np.any(warped_canvas != 0, axis=2)  # True where at least one channel is non-black
        
        for c in range(3):  # BGR channels
            frame[:, :, c][mask] = warped_canvas[:, :, c][mask]

    cv2.imshow("dbhs", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break