from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]

def load_model(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    return model

def get_vehicle_boxes(results):
    boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id in VEHICLE_CLASSES and confidence > 0.5:
                coords = box.xyxy[0].tolist()
                boxes.append(coords)
    return boxes

def calculate_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1

    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    inter_area = inter_width * inter_height
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def detect_accident(boxes, iou_threshold=0.1):
    if len(boxes) < 2:
        return False, []

    accident_pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                accident_pairs.append((boxes[i], boxes[j]))

    is_accident = len(accident_pairs) > 0
    return is_accident, accident_pairs