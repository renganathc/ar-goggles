from ultralytics import YOLO
import cv2 as cv

def detection(frame):
    model = YOLO("yolo11n.pt")

    results = model(frame, conf = 0.25)
    result_frame = results[0].plot()

    return result_frame

def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        resultant_frame = detection(frame)

        cv.imshow("Object Detection", resultant_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()