import cv2
import numpy as np
from ultralytics import YOLO

class PetDetector:
    def __init__(self, model_path="comp_vis_general/detector_pets/weights/yolo26n.pt"):
        # We use the nano model for high-speed home-server deployment
        self.model = YOLO(model_path)
        # COCO class IDs: 15 is cat, 16 is dog
        self.target_classes = [15, 16] 

    def predict(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Inference with class filtering
        results = self.model(img, classes=self.target_classes)[0]
        
        detections = []
        for box in results.boxes:
            detections.append({
                "species": results.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                "coordinates": box.xyxy[0].tolist()
            })
            
        return detections
        