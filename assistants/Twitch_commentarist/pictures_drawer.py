# OpenCV
import cv2
import numpy as np
import os
import random
from threading import Thread

def display_window(self):
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.window_name, 1024, 1024)
    current_image = None
    last_state = False
    
    while True:
        if self.audio_to_reproduce[0] and not last_state:
            # Seleccionar nueva imagen aleatoria solo cuando comienza el audio
            images = [f for f in os.listdir(self.image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                random_image = random.choice(images)
                try:
                    current_image = cv2.imread(os.path.join(self.image_directory, random_image), cv2.IMREAD_UNCHANGED)
                    # Si la imagen tiene canal alfa (transparencia)
                    if current_image is not None and current_image.shape[-1] == 4:
                        # Convertir a BGR eliminando la transparencia
                        alpha_channel = current_image[:, :, 3]
                        rgb_channels = current_image[:, :, :3]
                        
                        # Crear un fondo verde (BGR: 0,255,0)
                        green_background = np.zeros_like(rgb_channels, dtype=np.uint8)
                        green_background[:] = (0, 255, 0)
                        
                        # Crear máscara del canal alfa
                        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
                        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
                        
                        # Combinar imagen con fondo verde
                        current_image = (rgb_channels.astype(np.float32) * alpha_factor + 
                                        green_background.astype(np.float32) * (1 - alpha_factor))
                        current_image = current_image.astype(np.uint8)
                    
                    if current_image is not None and current_image.size > 0:
                        cv2.imshow(self.window_name, current_image)
                    else:
                        # Si hay error al cargar la imagen, mostrar pantalla verde
                        green_self = np.zeros((720, 1280, 3), dtype=np.uint8)
                        green_self[:] = (0, 255, 0)
                        cv2.imshow(self.window_name, green_self)
                except Exception as e:
                    print(f"Error al cargar la imagen: {e}")
                    green_self = np.zeros((720, 1280, 3), dtype=np.uint8)
                    green_self[:] = (0, 255, 0)
                    cv2.imshow(self.window_name, green_self)
        elif self.audio_to_reproduce[0] and last_state:
            # Mantener la imagen actual si es válida
            if current_image is not None and current_image.size > 0:
                cv2.imshow(self.window_name, current_image)
            else:
                green_self = np.zeros((720, 1280, 3), dtype=np.uint8)
                green_self[:] = (0, 255, 0)
                cv2.imshow(self.window_name, green_self)
        else:
            # Mostrar pantalla verde
            green_self = np.zeros((720, 1280, 3), dtype=np.uint8)
            green_self[:] = (0, 255, 0)
            cv2.imshow(self.window_name, green_self)
        
        last_state = self.audio_to_reproduce[0]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break