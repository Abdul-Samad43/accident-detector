from ultralytics import YOLO
import math

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

def load_model():
    return YOLO("yolo11n.pt")


def get_vehicle_boxes(model, image_path):
    results = model(image_path, conf=0.20, verbose=False)

    boxes = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]

        if label in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            boxes.append({
                "label": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })

    return boxes


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def center_distance(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    c1x = (x1 + x2) / 2
    c1y = (y1 + y2) / 2
    c2x = (a1 + a2) / 2
    c2y = (b1 + b2) / 2

    return math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)


def detect_accident(boxes):
    if len(boxes) < 2:
        return False, []

    accident_pairs = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i]["box"]
            box2 = boxes[j]["box"]

            iou = calculate_iou(box1, box2)
            distance = center_distance(box1, box2)

            x1, y1, x2, y2 = box1
            a1, b1, a2, b2 = box2

            w1 = x2 - x1
            h1 = y2 - y1
            w2 = a2 - a1
            h2 = b2 - b1

            size_limit = (w1 + h1 + w2 + h2) * 0.25

            if iou > 0.03 or distance < size_limit:
                accident_pairs.append((boxes[i], boxes[j]))

    return len(accident_pairs) > 0, accident_pairs