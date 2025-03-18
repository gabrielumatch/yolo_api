import os
import pytest
from io import BytesIO
from unittest.mock import patch
from werkzeug.datastructures import FileStorage

@pytest.mark.integration
class TestAPI:
    def test_index_page(self, client):
        """Test the index page loads correctly."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'YOLO Object Detection' in response.data

    def test_detect_endpoint_no_file(self, client):
        """Test detect endpoint without file."""
        response = client.post('/api/detect')
        assert response.status_code == 400
        assert b'No file part' in response.data

    def test_detect_endpoint_empty_file(self, client):
        """Test detect endpoint with empty file."""
        data = {}
        data['file'] = (BytesIO(b''), '')
        response = client.post('/api/detect', data=data)
        assert response.status_code == 400
        assert b'No selected file' in response.data

    def test_detect_endpoint_invalid_file(self, client):
        """Test detect endpoint with invalid file type."""
        data = {}
        data['file'] = (BytesIO(b'not an image'), 'test.txt')
        response = client.post('/api/detect', data=data)
        assert response.status_code == 400
        assert b'File type not allowed' in response.data

    def test_detect_endpoint_success(self, client, sample_image):
        """Test successful detection."""
        mock_results = [
            {
                'class': 'person',
                'confidence': 0.95,
                'bbox': [100, 100, 200, 200]
            }
        ]

        with open(sample_image, 'rb') as img, \
             patch('app.utils.yolo_detector.YOLODetector.detect', return_value=mock_results):
            
            data = {}
            data['file'] = (img, 'test.png')
            response = client.post('/api/detect', data=data)
            
            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data['success'] is True
            assert 'detections' in json_data
            assert isinstance(json_data['detections'], list)
            assert len(json_data['detections']) == 1
            assert json_data['detections'][0]['class'] == 'person'

    def test_detect_endpoint_large_file(self, client):
        """Test detect endpoint with file exceeding size limit."""
        # Create a large file (17MB)
        large_data = b'0' * (17 * 1024 * 1024)
        data = {}
        data['file'] = (BytesIO(large_data), 'large.png')
        
        # Set a smaller MAX_CONTENT_LENGTH for testing
        client.application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
        
        response = client.post('/api/detect', data=data)
        assert response.status_code == 413  # Request Entity Too Large

    def test_upload_folder_creation(self, app, client, sample_image):
        """Test upload folder is created if it doesn't exist."""
        # Remove upload folder if it exists
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            os.rmdir(app.config['UPLOAD_FOLDER'])
        
        mock_results = [
            {
                'class': 'person',
                'confidence': 0.95,
                'bbox': [100, 100, 200, 200]
            }
        ]

        with open(sample_image, 'rb') as img, \
             patch('app.utils.yolo_detector.YOLODetector.detect', return_value=mock_results):
            
            data = {}
            data['file'] = (img, 'test.png')
            response = client.post('/api/detect', data=data)
            
            assert response.status_code == 200
            assert os.path.exists(app.config['UPLOAD_FOLDER']) 