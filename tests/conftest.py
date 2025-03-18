import os
import pytest
import tempfile
from flask import Flask
from app import create_app
from app.utils.yolo_detector import YOLODetector

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Create a temporary file to store uploads
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_app('testing')
        app.config['UPLOAD_FOLDER'] = temp_dir
        app.config['TESTING'] = True
        yield app

@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a test runner for the app's CLI commands."""
    return app.test_cli_runner()

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a 1x1 pixel PNG image
    image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe2\x21\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(image_data)
        return f.name

@pytest.fixture
def mock_yolo_detector(mocker):
    """Mock the YOLO detector for testing."""
    mock_detector = mocker.patch('app.utils.yolo_detector.YOLODetector')
    mock_detector.return_value.detect.return_value = [
        {
            'class': 'person',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200]
        }
    ]
    return mock_detector 