from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image
app = Flask(__name__)


# Define parameters directly in the code
MODEL_NAME = "custom_model_lite"  # Cambia esto al nombre de tu directorio del modelo
GRAPH_NAME = "detect.tflite"  # Nombre del archivo del modelo .tflite
LABELMAP_NAME = 'custom_model_lite/labelmap.txt'  # Nombre del archivo de etiquetas
min_conf_threshold = 0.1  # Umbral de confianza mínimo


# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=os.path.join(MODEL_NAME, GRAPH_NAME))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(LABELMAP_NAME, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del (labels[0])


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image']

        # Decodificar la imagen de base64
        image = base64.b64decode(image_data)

        # Abrir la imagen usando PIL
        img = Image.open(io.BytesIO(image))
        img = img.convert('RGB')  # Asegúrate de que la imagen esté en RGB

        # Redimensionar a las dimensiones necesarias para el modelo
        img = img.resize((320, 320))  # Cambia a las dimensiones requeridas por tu modelo

        # Convertir la imagen a un arreglo NumPy
        img_array = np.array(img)

        # Normalizar la imagen
        img_array = img_array.astype(np.float32) / 255.0

        # Agregar la dimensión de lote
        img_array = np.expand_dims(img_array, axis=0)  # Cambia la forma a (1, 320, 320, 3)

        # Pasar la imagen al modelo
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Realizar la inferencia
        interpreter.invoke()

        # Obtener los resultados
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_boxes = interpreter.get_tensor(output_details[1]['index'])
        print("output_boxes")
        print(output_boxes)
        # Procesar los resultados
        results = []
        for i in range(len(output_data[0])):
            score = output_data[0][i]
            if score >= min_conf_threshold:
                ymin, xmin, ymax, xmax = output_boxes[0][i]
                result = {
                    'class': labels[i],
                    'score': float(score),  # Asegúrate de convertir a float
                    'box': [float(ymin), float(xmin), float(ymax), float(xmax)]  # Convertir las coordenadas a float
                }
                results.append(result)

        return jsonify(results)


    except Exception as e:
        print(f'Error en la detección: {str(e)}')
        return jsonify({"error": "Error al procesar la imagen"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=90)