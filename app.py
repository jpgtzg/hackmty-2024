from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": ["http://localhost:8081", "http://10.22.139.44:5000"]}})

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files) # imprimir los archivos en la solicitud
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo.'}), 400

    file = request.files['file']
    print(file) # imprimir información del archivo
    if file.filename == '':
        return jsonify({'error': 'El archivo no tiene un nombre de archivo válido.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('images', filename)
    print(filepath) # imprimir la ruta del archivo
    file.save(filepath)

    subprocess.run(["python", "final.py", filepath])

    return jsonify({'success': 'Archivo subido y procesado correctamente.'}), 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)