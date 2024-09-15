import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2 as cv
import torch
import openvino as ov

# Fetch `notebook_utils` module
# Estas son tools hechas por OpenVINO para facilitar el uso de sus modelos, esto genera un archivo llamado `notebook_utils.py`
import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)


# Importar las herramientas necesarias de OpenVINO
from notebook_utils import download_file, VideoPlayer, device_widget, quantization_widget


# Modelos de PyTorch (esto sirve para hacer la comparativa con OpenVINO)
IMAGE_PATH = Path("images/hoja.jpg")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
det_model = YOLO("models/best.pt")
label_map = det_model.model.names

# Seleccionamos el CPU como dispositivo para hacer la inferencia
device = device_widget('CPU')

# Inicializamos un OpenVINO Core
core = ov.Core()

# Modelo de OpenVINO (este fue primero convertido de PyTorch a ONNX y luego a OpenVINO)
det_ov_model = core.read_model("models/best.xml")

# Detectamos objetos en la imagen con OpenVINO
ov_config = {}
if device.value != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
det_compiled_model = core.compile_model(det_ov_model, device.value, ov_config)

# Definimos una función para hacer inferencias con OpenVINO
def infer(*args):
    result = det_compiled_model(args)
    return torch.from_numpy(result[0])


# Hacer la inferencia con el modelo YOLO
res = det_model(IMAGE_PATH)

# Procesar los resultados para guardarlos en un archivo JSON
detections = []
for detection in res[0].boxes:  # Recorremos las detecciones
    box = detection.xyxy.tolist()[0]  # Coordenadas de la caja delimitadora (x1, y1, x2, y2)
    score = detection.conf.tolist()[0]  # Confianza de la predicción (accedemos al primer valor)
    class_id = detection.cls.tolist()[0]  # ID de la clase (accedemos al primer valor)
    label = label_map[int(class_id)]  # Nombre de la clase a partir del mapa de etiquetas

    detections.append({
        "class": label,
        "confidence": score,
        "box": box
    })

# Guardar las detecciones en un archivo JSON
output_path = Path("json/detections.json")
with open(output_path, "w") as f:
    json.dump(detections, f, indent=4)

print(f"Detecciones guardadas en {output_path}")

# Mostrar la imagen con las detecciones
Image.fromarray(res[0].plot()[:, :, ::-1])

# Añadir código OpenCV para ver la imagen en una ventana
cv.imshow("Image", res[0].plot()[:, :, ::-1])
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar la imagen con las detecciones
output_image_path = Path("images/detections.jpg")
res[0].save(output_image_path)
print(f"Imagen con detecciones guardada en {output_image_path}")

