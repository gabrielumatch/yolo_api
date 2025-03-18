import os
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize the YOLO detector
        
        Args:
            model_name: The name of the YOLO model to use
        """
        # Load the YOLO model
        self.model = YOLO(model_name)
        
    def detect(self, image_path, conf=0.25):
        """
        Detect objects in an image
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            
        Returns:
            List of detection results with class, confidence, and bounding box
        """
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        # Run inference
        results = self.model(image_path, conf=conf)
        
        # Process results
        detections = []
        
        # For the first image result (assumes a single image was passed)
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get box coordinates (convert to int for JSON serialization)
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                
                # Get confidence
                confidence = float(box.conf[0])
                
                # Get class name
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detections 