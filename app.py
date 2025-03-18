import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import time
from app.utils.yolo_detector import YOLODetector

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the YOLO detector
detector = YOLODetector()

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
        
        # Run detection on the image and save the processed image
        results = detector.detect(filepath)
        
        # Generate URL for the processed image
        image_url = url_for('static', filename=f'uploads/{filename}')
        
        # Return detection results with image URL
        return jsonify({
            'success': True,
            'result_image': image_url,
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
        results = detector.detect(filepath)
        
        # Return only the detection results without the image
        return jsonify({
            'success': True,
            'detections': results
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 