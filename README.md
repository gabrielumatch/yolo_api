# YOLO Object Detection API

A Flask-based API for object detection using YOLOv8. This API provides both a web interface and REST endpoints for object detection.

## Examples

Here are some examples of object detection using our API:

### Example 1: Person Detection

![Person Detection](https://github.com/gabrielumatch/yolo_api/blob/main/assets/1.png)

### Example 2: Multiple Objects Detection

![Person Detection](https://github.com/gabrielumatch/yolo_api/blob/main/assets/2.png)

## Requirements

- Python 3.10
- CUDA-capable GPU (recommended)
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space
- Docker and Docker Compose (for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yolo_api.git
cd yolo_api
```

2. Create and activate a virtual environment:

```bash
python -m venv venv310
.\venv310\Scripts\activate  # On Windows
source venv310/bin/activate  # On Linux/Mac
```

3. Install PyTorch with CUDA support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install other dependencies:

```bash
pip install -r requirements.txt
```

### Docker Installation

1. Ensure you have Docker and Docker Compose installed on your system.

2. Clone the repository:

```bash
git clone https://github.com/yourusername/yolo_api.git
cd yolo_api
```

3. Build and run the container:

```bash
docker-compose up --build
```

The API will be available at http://localhost:5000

## Testing

The project includes a comprehensive test suite using pytest. To run the tests:

1. Install test dependencies:

```bash
pip install -r requirements.txt
```

2. Run all tests:

```bash
pytest
```

3. Run specific test categories:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests without slow tests
pytest -m "not slow"
```

4. Generate coverage report:

```bash
pytest --cov=app --cov-report=html
```

The coverage report will be generated in the `htmlcov` directory.

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the API endpoints and YOLO integration
- **Slow Tests**: Tests that require actual model inference (can be skipped with `-m "not slow"`)

## Configuration

The application supports different environments (development, production, testing) through environment variables and configuration files.

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit the `.env` file with your settings:

```env
# Flask Settings
SECRET_KEY=your-secret-key
FLASK_ENV=development  # Options: development, production, testing

# Server Settings
HOST=0.0.0.0
PORT=5000

# YOLO Settings
YOLO_MODEL=yolov8n.pt
YOLO_CONF=0.25

# Upload Settings
UPLOAD_FOLDER=app/static/uploads
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
```

### Environment-Specific Settings

- **Development**: Debug mode enabled, local upload folder
- **Production**: Debug mode disabled, secure upload folder, optimized settings
- **Testing**: Debug mode enabled, temporary upload folder

## Usage

### Running the Application

#### Local Development

1. Development mode:

```bash
set FLASK_ENV=development  # On Windows
export FLASK_ENV=development  # On Linux/Mac
python app.py
```

2. Production mode:

```bash
set FLASK_ENV=production  # On Windows
export FLASK_ENV=production  # On Linux/Mac
python app.py
```

3. Testing mode:

```bash
set FLASK_ENV=testing  # On Windows
export FLASK_ENV=testing  # On Linux/Mac
python app.py
```

#### Docker Deployment

1. Start the container:

```bash
docker-compose up
```

2. Stop the container:

```bash
docker-compose down
```

3. View logs:

```bash
docker-compose logs -f
```

### Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Upload an image using the web interface
3. View detection results with bounding boxes and confidence scores

### API Usage

Send a POST request to `http://localhost:5000/api/detect` with an image file:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:5000/api/detect
```

The API will return a JSON response containing:

- Object class
- Confidence score
- Bounding box coordinates

## Model Information

This API uses YOLOv8n (nano) model, which is:

- Efficient for cheaper instances
- Fast inference times
- Good balance of speed and accuracy
- Supports 80 different classes of objects

## Customization

### Different YOLO Models

You can use different YOLO models by changing the `YOLO_MODEL` in your `.env` file:

- `yolov8n.pt`: Nano (fastest, smallest)
- `yolov8s.pt`: Small
- `yolov8m.pt`: Medium
- `yolov8l.pt`: Large
- `yolov8x.pt`: XLarge (slowest, most accurate)

### Confidence Threshold

Adjust the confidence threshold in your `.env` file:

```env
YOLO_CONF=0.25  # Default value, range: 0.0 to 1.0
```

## Troubleshooting

1. **CUDA Errors**:

   - Ensure PyTorch is installed with CUDA support
   - Check GPU drivers are up to date
   - Verify CUDA toolkit is installed

2. **Memory Issues**:

   - Reduce image size
   - Use a smaller YOLO model
   - Close other memory-intensive applications

3. **Upload Errors**:

   - Check file size (max 16MB)
   - Verify file type (png, jpg, jpeg, gif)
   - Ensure upload folder has write permissions

4. **Docker Issues**:

   - Ensure NVIDIA Container Toolkit is installed for GPU support
   - Check Docker service is running
   - Verify port 5000 is not in use
   - Check container logs for detailed error messages

5. **Test Issues**:
   - Ensure all dependencies are installed
   - Check pytest configuration
   - Verify test environment variables
   - Check coverage report for uncovered code

## Cloud Deployment

### AWS EC2 with GPU

1. Launch an EC2 instance with GPU support (e.g., g4dn.xlarge)
2. Install Docker and NVIDIA Container Toolkit
3. Clone the repository
4. Run with docker-compose:

```bash
docker-compose up -d
```

### Google Cloud Platform

1. Create a Compute Engine instance with GPU
2. Install Docker and NVIDIA Container Toolkit
3. Clone the repository
4. Run with docker-compose:

```bash
docker-compose up -d
```

### Azure

1. Create a VM with GPU support
2. Install Docker and NVIDIA Container Toolkit
3. Clone the repository
4. Run with docker-compose:

```bash
docker-compose up -d
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
