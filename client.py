from ultralytics import YOLO
from datetime import datetime
import socketio
import base64
import cv2

sio = socketio.Client()

@sio.event
def connect():
    print("Connected to the server")

@sio.event
def disconnect():
    print("Disconnected from the server.")

sio.connect('http://localhost:5050')

def send_detection(image, timestamp):
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    sio.emit('detection_result', {
        'image': img_base64,
        'time': timestamp
    })

def main():
    model = YOLO("yolo26s.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results = model(frame, verbose=False)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        plotted_img = results[0].plot()
        send_detection(plotted_img, timestamp)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
