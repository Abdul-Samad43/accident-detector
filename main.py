from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os 
from accident_detector import load_model, get_vehicle_boxes, detect_accident
from PIL import Image
import io

app  = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
from PIL import Image
import io

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = image.resize((640, 640))
    
    file_path = f"temp_{file.filename}"
    image.save(file_path)
    
    results = model(file_path)
model = load_model("yolo11n.pt")
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((640, 640))
    file_path = f"temp_{file.filename}"
    image.save(file_path)
    results = model(file_path)
    boxes = get_vehicle_boxes(results)
    is_accident, pairs = detect_accident(boxes)
    os.remove(file_path)
    return {
        "vehicles_detected": len(boxes),
        "accident_detected": is_accident,
        "accident_pairs": len(pairs)
    }
    
@app.get("/")
def home():
    return {"status": "backend running"}