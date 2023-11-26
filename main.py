import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

# Utiliza el modelo entrenado YOLOv8tiny
model = YOLO('../best_ultimoModelo.pt')

# Ejecuta el seguimiento de objetos en el video "test.mp4"
results = model.track(source='testFinal1.mp4', show=True)