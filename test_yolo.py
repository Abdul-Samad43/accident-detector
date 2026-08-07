from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("yolo11n.pt")

# Test on sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Draw detection boxes
result_image = results[0].plot()

# Show result
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()