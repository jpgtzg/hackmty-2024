import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import openvino as ov
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
import base64
import requests

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": ["10.22.130.241:8081", "10.22.130.241:5001"]}})

# Descarga de herramientas de OpenVINO
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py", "w").write(r.text)
from notebook_utils import download_file, device_widget, quantization_widget

# Procesar la imagen y aplicar detección
def process_image(filepath):
    IMAGE_PATH = Path(filepath)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    det_model = YOLO("models/best.pt")
    label_map = det_model.model.names

    # Inicializar OpenVINO Core
    core = ov.Core()
    det_ov_model = core.read_model("models/best.xml")
    device = device_widget('CPU')

    # Compilar el modelo en OpenVINO
    ov_config = {}
    if device.value != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, device.value, ov_config)

    # Inferencia con YOLO y OpenVINO
    res = det_model(IMAGE_PATH)

    # Procesar resultados y generar archivo JSON
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

    # Guardar detecciones en JSON
    output_path = Path("json/detections.json")
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=4)

    # Guardar la imagen con detecciones
    output_image_path = Path("images/detections.jpg")
    res[0].save(output_image_path)

    # Convertir la imagen a base64 para enviarla
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

    # Guardar el archivo temporalmente
    filepath = Path("uploaded_image.jpg")
    file.save(filepath)

    try:
        # Procesar la imagen
        processed_image_base64, detections = process_image(filepath)

        # Retornar la imagen procesada en formato base64 y el JSON con las detecciones
        return jsonify({
            'processed_image': processed_image_base64,
            'detections': detections
        })
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return jsonify({'error': 'Error al procesar la imagen'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
