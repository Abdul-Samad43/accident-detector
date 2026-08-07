from accident_detector import load_model, get_vehicle_boxes, detect_accident

model = load_model("yolo11n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
boxes = get_vehicle_boxes(results)
print("Detected Vehicle Boxes:", len(boxes))
print("boxes:", boxes)

is_accident, pairs = detect_accident(boxes)
print("Accident detected:", is_accident)
print("Accident pairs:", pairs)
 