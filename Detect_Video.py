import cv2
import torch
from ultralytics import YOLO
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA Available:", torch.cuda.is_available())
print("Using device:", device)


model = YOLO("yolov8-weights/yolov8n.pt")
model.to(device)


video_path = "videos/input.mp4"
cap = cv2.VideoCapture(video_path)


width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)


os.makedirs("videos", exist_ok=True)
out = cv2.VideoWriter("videos/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break


    results = model(frame, device='0')  # use GPU ID 0



    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    out.write(frame)
    cv2.imshow("YOLOv8 Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()



