import pytest
import torch
from unittest.mock import MagicMock, patch
from app.utils.yolo_detector import YOLODetector

@pytest.mark.unit
class TestYOLODetector:
    def test_init(self):
        """Test YOLO detector initialization."""
        with patch('app.utils.yolo_detector.YOLO') as mock_yolo:
            detector = YOLODetector('yolov8n.pt')
            assert detector.device == ('cuda' if torch.cuda.is_available() else 'cpu')
            mock_yolo.assert_called_once_with('yolov8n.pt')

    def test_detect_with_mock(self, sample_image):
        """Test detection with mocked YOLO model."""
        mock_results = MagicMock()
        mock_results.boxes.data.tolist.return_value = [
            [100, 100, 200, 200, 0.95, 0]  # x1, y1, x2, y2, conf, class_id
        ]
        mock_results.names = {0: 'person'}

        with patch('app.utils.yolo_detector.YOLO') as mock_yolo:
            mock_model = mock_yolo.return_value
            mock_model.return_value = [mock_results]
            
            detector = YOLODetector('yolov8n.pt')
            results = detector.detect(sample_image)
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]['class'] == 'person'
            assert results[0]['confidence'] == 0.95
            assert results[0]['bbox'] == [100.0, 100.0, 200.0, 200.0]

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
        with pytest.raises(FileNotFoundError):
            detector.detect('nonexistent_image.jpg')

    def test_different_confidence_threshold(self, sample_image):
        """Test detection with different confidence threshold."""
        mock_results = MagicMock()
        mock_results.boxes.data.tolist.return_value = [
            [100, 100, 200, 200, 0.95, 0]  # x1, y1, x2, y2, conf, class_id
        ]
        mock_results.names = {0: 'person'}

        with patch('app.utils.yolo_detector.YOLO') as mock_yolo:
            mock_model = mock_yolo.return_value
            mock_model.return_value = [mock_results]
            
            detector = YOLODetector('yolov8n.pt')
            results = detector.detect(sample_image, conf=0.5)
            
            assert isinstance(results, list)
            assert len(results) == 1
            mock_model.assert_called_once_with(sample_image, conf=0.5) 