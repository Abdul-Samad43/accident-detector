from ultralytics import YOLO

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

def load_model():
    return YOLO("yolo11n.pt")


def get_vehicle_boxes(model, image_path):
    results = model(image_path, conf=0.50, verbose=False)

    boxes = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]

        if label in VEHICLE_CLASSES and conf >= 0.50:
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

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def detect_accident(boxes):
    if len(boxes) < 2:
        return False, []

    accident_pairs = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i]["box"], boxes[j]["box"])

            if iou > 0.20:
                accident_pairs.append((boxes[i], boxes[j]))

    if len(accident_pairs) > 0:
        return True, accident_pairs

    return False, []