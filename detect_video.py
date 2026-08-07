import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
input_path = Path("input_video.mp4")
output_path = Path("output_video.mp4")

# Check input video exists
if not input_path.exists():
    print("Error: input_video.mp4 not found in project folder.")
    exit()

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture(str(input_path))

if not cap.isOpened():
    print("Error: Video cannot be opened.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Make sure frame has 3 color channels
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # YOLO detection on actual frame
    results = model(frame, conf=0.4, verbose=False)

    # Draw boxes
    annotated_frame = results[0].plot()

    # Make sure output frame size is same as video writer
    annotated_frame = cv2.resize(annotated_frame, (width, height))

    # Make sure output has 3 channels
    if len(annotated_frame.shape) == 2:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)

    out.write(annotated_frame)

    print(f"Frame {frame_count} processed")

cap.release()
out.release()

print("Done! output_video.mp4 saved.")