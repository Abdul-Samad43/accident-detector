from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from accident_detector import load_model, get_vehicle_boxes, detect_accident
from PIL import Image
import os
import io
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((640, 640))

        file_path = f"temp_{uuid.uuid4().hex}.jpg"
        image.save(file_path)

        boxes = get_vehicle_boxes(model, file_path)
        is_accident, pairs = detect_accident(boxes)

        os.remove(file_path)

        return {
            "accident_detected": is_accident,
            "vehicles_detected": len(boxes)
        }

    except Exception as e:
        return {
            "error": str(e),
            "accident_detected": False,
            "vehicles_detected": 0
        }