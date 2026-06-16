import ultralytics
import cv2

model = ultralytics.YOLO("yolo11n.pt")
print(model.names)