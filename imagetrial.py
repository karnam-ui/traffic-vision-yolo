from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")
img = cv2.imread("corridor-research-institute-aggarwal-transit-corridor-corridor_b470e230-1973-11e9-b6e9-9c4bb39de67f.jpg")
results = model(img, conf=0.50)
annotated = results[0].plot()

cv2.imshow("Annotated Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()