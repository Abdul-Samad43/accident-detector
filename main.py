from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from accident_detector import load_model, get_vehicle_boxes, detect_accident
from PIL import Image
import os, io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("yolo11n.pt")

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((640, 640))
        file_path = f"temp_{file.filename}"
        image.save(file_path)

        results = model(file_path)
        boxes = get_vehicle_boxes(results)

        print(f"Total boxes: {len(boxes)}")

        is_accident, pairs = detect_accident(boxes)
        os.remove(file_path)

        return {
            "vehicles_detected": len(boxes),
            "accident_detected": is_accident,
            "accident_pairs": len(pairs)
        }
    except Exception as e:
        return {"error": str(e)}