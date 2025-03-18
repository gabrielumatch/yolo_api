import pytest
import torch
from app.utils.yolo_detector import YOLODetector

@pytest.mark.unit
class TestYOLODetector:
    def test_init(self):
        """Test YOLO detector initialization."""
        detector = YOLODetector('yolov8n.pt')
        assert detector.model is not None
        assert detector.device == ('cuda' if torch.cuda.is_available() else 'cpu')

    def test_detect_with_mock(self, mock_yolo_detector, sample_image):
        """Test detection with mocked YOLO model."""
        detector = YOLODetector('yolov8n.pt')
        results = detector.detect(sample_image)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(det, dict) for det in results)
        assert all('class' in det for det in results)
        assert all('confidence' in det for det in results)
        assert all('bbox' in det for det in results)

    @pytest.mark.slow
    def test_detect_with_real_model(self, sample_image):
        """Test detection with real YOLO model (slow test)."""
        detector = YOLODetector('yolov8n.pt')
        results = detector.detect(sample_image)
        
        assert isinstance(results, list)
        assert all(isinstance(det, dict) for det in results)
        assert all('class' in det for det in results)
        assert all('confidence' in det for det in results)
        assert all('bbox' in det for det in results)

    def test_invalid_image(self):
        """Test detection with invalid image."""
        detector = YOLODetector('yolov8n.pt')
        with pytest.raises(Exception):
            detector.detect('nonexistent_image.jpg')

    def test_different_confidence_threshold(self, mock_yolo_detector, sample_image):
        """Test detection with different confidence threshold."""
        detector = YOLODetector('yolov8n.pt')
        results = detector.detect(sample_image, conf=0.5)
        assert isinstance(results, list) 