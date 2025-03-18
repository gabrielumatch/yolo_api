import os
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO detector with specified model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, image_path, conf=0.25):
        """
        Detect objects in an image.
        
        Args:
            image_path (str): Path to the image file
            conf (float): Confidence threshold (0-1)
            
        Returns:
            list: List of dictionaries containing detection results
                 Each dictionary contains 'class', 'confidence', and 'bbox'
        """
        try:
            # Run inference
            results = self.model(image_path, conf=conf)[0]
            
            # Process results
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                detection = {
                    'class': results.names[int(class_id)],
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            raise 