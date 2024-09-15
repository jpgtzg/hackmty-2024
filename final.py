from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import openvino as ov
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from io import BytesIO
import base64
import requests
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": ["http://10.22.130.241:8081", "http://10.22.130.241:5001"]}})

# Download OpenVINO utilities to local environment
r = requests.get("https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
open("notebook_utils.py", "w").write(r.text)
from notebook_utils import device_widget

# Ensure necessary directories exist
Path("models").mkdir(exist_ok=True)
Path("json").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)

# Serve static files
@app.route('/json/<path:filename>', methods=['GET'])
def serve_json(filename):
    filepath = Path(f"json/{filename}")
    if filepath.exists():
        print(f"Serving JSON file: {filename}")  # Debug info
        return send_from_directory('json', filename)
    else:
        print(f"File not found: {filename}")  # Debug info
        return jsonify({'error': 'File not found'}), 404

@app.route('/images/<path:filename>', methods=['GET'])
def serve_images(filename):
    filepath = Path(f"images/{filename}")
    if filepath.exists():
        print(f"Serving image: {filename}")  # Debug info
        return send_from_directory('images', filename)
    else:
        print(f"Image not found: {filename}")  # Debug info
        return jsonify({'error': 'Image not found'}), 404

# Process image and apply detection
def process_image(filepath):
    IMAGE_PATH = Path(filepath)
    det_model = YOLO("models/best.pt")
    label_map = det_model.model.names

    # Initialize OpenVINO Core
    core = ov.Core()
    det_ov_model = core.read_model("models/best.xml")
    device = device_widget('CPU')

    # Compile the model in OpenVINO
    ov_config = {}
    if device.value != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, device.value, ov_config)

    # Inference with YOLO and OpenVINO
    res = det_model(IMAGE_PATH)

    # Process results and generate JSON file
    detections = []
    for detection in res[0].boxes:
        box = detection.xyxy.tolist()[0]
        score = detection.conf.tolist()[0]
        class_id = detection.cls.tolist()[0]
        label = label_map[int(class_id)]

        detections.append({
            "class": label,
            "confidence": score,
            "box": box
        })

    # Save detections in JSON
    output_path = Path("json/detections.json")
    print(f"Saving detections to: {output_path}")  # Debug info
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=4)

    # Save the image with detections
    output_image_path = Path("images/detections.jpg")
    print(f"Saving image to: {output_image_path}")  # Debug info
    res[0].save(output_image_path)

    # Convert the image to base64 to send it
    buffered = BytesIO()
    img = Image.open(output_image_path)
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str, detections

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return jsonify({'error': 'No se ha enviado ningún archivo'}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400

    filepath = Path("uploaded_image.jpg")
    file.save(filepath)

    try:
        processed_image_base64, detections = process_image(filepath)

        # Save paths to return to the front-end
        detections_path = "json/detections.json"
        image_path = "images/detections.jpg"

        # Verifica la existencia de los archivos antes de responder
        if not Path(detections_path).exists():
            print(f"JSON file not found: {detections_path}")
        if not Path(image_path).exists():
            print(f"Image file not found: {image_path}")

        return jsonify({
            'processed_image': processed_image_base64,
            'detections': detections,
            'detections_path': detections_path,
            'image_path': image_path
        })
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return jsonify({'error': 'Error al procesar la imagen'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)