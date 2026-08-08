from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os 
from accident_detector import load_model, get_vehicle_boxes, detect_accident

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
    
    # Upload ki hui image save karo
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # YOLO chalao
    results = model(file_path)
    
    # Vehicles nikalo
    boxes = get_vehicle_boxes(results)
    
    # Accident check karo
    is_accident, pairs = detect_accident(boxes)
    
    # Temp file delete karo
    os.remove(file_path)
    
    # Result return karo
    return {
        "vehicles_detected": len(boxes),
        "accident_detected": is_accident,
        "accident_pairs": len(pairs)
    }
    
@app.get("/")
def home():
    return {"status": "backend running"}