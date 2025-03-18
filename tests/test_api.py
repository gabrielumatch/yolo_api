import os
import pytest
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
        data = {'file': (b'', '')}
        response = client.post('/api/detect', data=data)
        assert response.status_code == 400
        assert b'No selected file' in response.data

    def test_detect_endpoint_invalid_file(self, client):
        """Test detect endpoint with invalid file type."""
        data = {'file': (b'not an image', 'test.txt')}
        response = client.post('/api/detect', data=data)
        assert response.status_code == 400
        assert b'File type not allowed' in response.data

    def test_detect_endpoint_success(self, client, sample_image, mock_yolo_detector):
        """Test successful detection."""
        with open(sample_image, 'rb') as f:
            data = {'file': (f, 'test.png')}
            response = client.post('/api/detect', data=data)
        
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['success'] is True
        assert 'detections' in json_data
        assert isinstance(json_data['detections'], list)
        assert len(json_data['detections']) > 0

    def test_detect_endpoint_large_file(self, client):
        """Test detect endpoint with file exceeding size limit."""
        # Create a large file (17MB)
        large_data = b'0' * (17 * 1024 * 1024)
        data = {'file': (large_data, 'large.png')}
        response = client.post('/api/detect', data=data)
        assert response.status_code == 413  # Request Entity Too Large

    def test_upload_folder_creation(self, app, client, sample_image):
        """Test upload folder is created if it doesn't exist."""
        # Remove upload folder if it exists
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            os.rmdir(app.config['UPLOAD_FOLDER'])
        
        # Make a request that will create the folder
        with open(sample_image, 'rb') as f:
            data = {'file': (f, 'test.png')}
            response = client.post('/api/detect', data=data)
        
        assert response.status_code == 200
        assert os.path.exists(app.config['UPLOAD_FOLDER']) 