from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/api/prediccion', methods=['POST'])
def predecir_genero_edad():
    data = request.json
    imagen_b64 = data['imagen']
    img_data = base64.b64decode(imagen_b64.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Aquí puedes insertar tu lógica de detección
    # y devolver, por ejemplo:
    return jsonify({'genero': 'Mujer', 'edad': '25 a 32'})

if __name__ == '__main__':
    app.run(debug=True)
