import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO detector with specified model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, image_path, conf=0.25):
        """
        Detect objects in an image and save the annotated image.
        
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
            
            # Get the original image
            img = cv2.imread(image_path)
            
            # Draw bounding boxes on the image
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_name = results.names[int(class_id)]
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class name and confidence
                label = f"{class_name} {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Save the annotated image
            cv2.imwrite(image_path, img)
            
            # Process results for return
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