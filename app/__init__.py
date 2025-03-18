# This file makes the app directory a Python package 

import os
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from app.config import config

def create_app(config_name='default'):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize the YOLO detector
    from app.utils.yolo_detector import YOLODetector
    detector = YOLODetector(app.config['YOLO_MODEL'])
    
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/detect', methods=['POST'])
    def detect_objects():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Run detection on the image
            results = detector.detect(filepath, conf=app.config['YOLO_CONF'])
            
            # Return detection results
            return jsonify({
                'success': True,
                'filename': filename,
                'detections': results
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/api/detect', methods=['POST'])
    def api_detect():
        """API endpoint that accepts an image and returns detection results as JSON"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Run detection on the image
            results = detector.detect(filepath, conf=app.config['YOLO_CONF'])
            
            # Return only the detection results without the image
            return jsonify({
                'success': True,
                'detections': results
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    return app 