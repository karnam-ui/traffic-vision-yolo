from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(r"/Users/suhaas/Downloads/4K Road traffic video for object detection and tracking - free download now.mp4")
#cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.65,imgsz=1280,iou=0.75)
    annotated = results[0].plot()
    cv2.imshow("Annotated Video", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()