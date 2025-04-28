# face_api/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .reconocimiento import algorithm
import os

@csrf_exempt
def reconocer(request):
    if request.method == 'POST':
        try:
            if 'imagen' not in request.FILES:
                return JsonResponse({'error': 'No se proporcionó ninguna imagen'}, status=400)

            imagen = request.FILES['imagen']
            nombre_temporal = 'temp_image.jpg'
            ruta_temporal = os.path.join(os.path.dirname(__file__), 'reconocimiento', nombre_temporal) # Guarda en la carpeta reconocimiento

            with open(ruta_temporal, 'wb+') as destination:
                for chunk in imagen.chunks():
                    destination.write(chunk)

            resultado = algorithm.detectar_genero_en_imagen(ruta_temporal)

            os.remove(ruta_temporal)

            return JsonResponse(resultado)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)
