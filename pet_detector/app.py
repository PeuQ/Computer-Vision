from fastapi import FastAPI, File, UploadFile
from model import PetDetector
import uvicorn

app = FastAPI(title="Pet Survailance API")
detector = PetDetector()

@app.get("/")
def health_check():
    return {"status": "ready"}

@app.post("/predict")
async def predict_ppe(file: UploadFile = File(...)):
    # Read the uploaded image
    image_bytes = await file.read()
    
    # Run the detection
    detections = detector.predict(image_bytes)
    
    return {
        "filename": file.filename,
        "detections": detections,
        "count": len(detections)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    