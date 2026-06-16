import cv2
from ultralytics import YOLO
cap = cv2.VideoCapture(r"/Users/suhaas/Downloads/4K Road traffic video for object detection and tracking - free download now.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    final = cv2.cvtColor(enhanced,cv2.COLOR_GRAY2BGR)
    model = YOLO("yolo11n.pt")
    results = model(final, conf=0.65,imgsz=1280,iou=0.75)
    annotated = results[0].plot()
    cv2.imshow("Annotated Video", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
