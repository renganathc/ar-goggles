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
            dst_pts[0] = corners[0][0][0]
            dst_pts[1] = corners[0][0][1]
            dst_pts[2] = corners[0][0][2]
            dst_pts[3] = corners[0][0][3]

            H, _ = cv2.findHomography(src_pts, dst_pts)
            warped_canvas = cv2.warpPerspective(canvas, H, (frame_width, frame_height))
            warped_canvas = cv2.resize(warped_canvas, (frame_width, frame_height))
            mask = np.any(warped_canvas != 0, axis=2)  # True where at least one channel is non-black

    if warped_canvas is not None:
        for c in range(3):  # BGR channels
            frame[:, :, c][mask] = warped_canvas[:, :, c][mask]

    cv2.imshow("dbhs", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
