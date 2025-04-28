# face_api/reconocimiento/algorithm.py
import cv2
import numpy as np
import os

# Cargar el clasificador Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar los modelos pre-entrenados para la detección de género (asegúrate de que las rutas sean correctas)
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(os.path.dirname(__file__), 'gender_deploy.prototxt'),
    os.path.join(os.path.dirname(__file__), 'gender_net.caffemodel')
)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

def detectar_genero_en_imagen(ruta_imagen):
    try:
        frame = cv2.imread(ruta_imagen)
        if frame is None:
            return {'error': 'No se pudo cargar la imagen.'}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if not faces.any():
            return {'genero': None, 'rostro_detectado': False}
        else:
            # Tomamos el primer rostro detectado
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            return {'genero': gender, 'rostro_detectado': True}

    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Ejemplo de cómo usar la función directamente
    resultado = detectar_genero_en_imagen('ruta/a/una/imagen.jpg')
    print(resultado)
