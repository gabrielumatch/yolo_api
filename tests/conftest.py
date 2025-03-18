import os
import pytest
import tempfile
import numpy as np
import cv2
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
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple test image (100x100 black image)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = os.path.join(temp_dir, 'test.png')
        cv2.imwrite(img_path, img)
        yield img_path

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