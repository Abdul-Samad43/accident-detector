from ultralytics import YOLO

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

def load_model():
    return YOLO("yolo11n.pt")


def get_vehicle_boxes(model, image_path):
    results = model(image_path, conf=0.35, verbose=False)
    boxes = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]

        if label in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "label": label,
                "confidence": round(conf, 2),
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

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)

    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union


def detect_accident(boxes):
    # single vehicle = no accident
    if len(boxes) < 2:
        return False, []

    accident_pairs = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i]["box"], boxes[j]["box"])

            # only clear overlap means accident
            if iou > 0.12:
                accident_pairs.append((boxes[i], boxes[j]))

    return len(accident_pairs) > 0, accident_pairs