# AI Assistant  
from components.ai_assistant import AI_Assistant
# Twitch
from twitchio import Message
from twitchio.ext import commands
import re
# Kokoro
from components.kokoro.kokoro_class import Kokoro
# Config
from configparser import ConfigParser
import pathlib
# OpenCV
import cv2
import numpy as np
import os
import random
from threading import Thread

# Get bot.py's father path
path = pathlib.Path(__file__).parent.resolve().__str__()

# Get fields from config.ini
config = ConfigParser()
config.read(os.path.join(path, "..", "..", "config.ini"))
account_fields = config["TWITCH_COMMENTARIST_CONFIG"]

# Create a commands.Bot class
class TwitchCommentarist(AI_Assistant, commands.Bot, Kokoro):

    def __init__(self): 
        
        lm_config = config["LM"]
        base_url = lm_config["base_url"]
        if "api_key" in lm_config:
            api_key = lm_config["api_key"]
            is_local = False
        else:
            api_key = None
            is_local = True
        model = lm_config["model"]
        
        # Initialize AI Assistant
        AI_Assistant.__init__(self, initial_prompt='''
        Tu propósito es responder a los comentarios de un directo de Twitch en español de España.
        Lo harás de manera humorística y con un tono sarcástico. Importante: no escribir NUNCA emotes ni caras.
        Por supuesto, deberás saludar a aquellos usuarios que se vayan incorporando y comentando por primera vez.
        Tus respuestas no deben ser extensas.
        Tu creador es andresitositoses y le harás caso en todo lo que te pida, en caso de que comente algo en el chat.
        ''',
        personalities_path=account_fields["personalities_path"],
        personality_name=account_fields["personality_name"],
        summarization_frequency=int(account_fields["summarization_frequency"]),
        auto_save=account_fields["auto_save"],
        lm_params=(base_url, api_key, model, is_local))
        # Initialize Twitch bot
        commands.Bot.__init__(self, token=account_fields["access_token"],
                         prefix=account_fields["prefix"],
                         initial_channels=[account_fields["channel_name"]],
                         client_secret=account_fields["client_secret"])
        # Initialize Kokoro
        Kokoro.__init__(self, voice=config["VOICE"]["voice"])
        
        # Inicializar variables para la ventana
        self.window_name = "AI Assistant"
        self.display_thread = None
        self.audio_to_reproduce = (False, -1)
        self.image_directory = os.path.join(path, "img/Perfectas")
        
        # Iniciar el thread de visualización del rostro
        self.display_thread = Thread(target=self._display_window)
        self.display_thread.daemon = True
        self.display_thread.start()
        self.audio_to_reproduce = (False, -1)

        # Reproducir audio de inicialización
        audio_arrays, duration_seconds = self.generate_audio("Inicialización del sistema completada.")
        self.audio_to_reproduce = (True, duration_seconds)
        self.reproduce_audio(audio_arrays)
        self.audio_to_reproduce = (False, -1)

    def _display_window(self):
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
                            green_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
                            green_screen[:] = (0, 255, 0)
                            cv2.imshow(self.window_name, green_screen)
                    except Exception as e:
                        print(f"Error al cargar la imagen: {e}")
                        green_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
                        green_screen[:] = (0, 255, 0)
                        cv2.imshow(self.window_name, green_screen)
            elif self.audio_to_reproduce[0] and last_state:
                # Mantener la imagen actual si es válida
                if current_image is not None and current_image.size > 0:
                    cv2.imshow(self.window_name, current_image)
                else:
                    green_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
                    green_screen[:] = (0, 255, 0)
                    cv2.imshow(self.window_name, green_screen)
            else:
                # Mostrar pantalla verde
                green_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
                green_screen[:] = (0, 255, 0)
                cv2.imshow(self.window_name, green_screen)
            
            last_state = self.audio_to_reproduce[0]
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    async def event_message(self, message: Message):
        'Display messages on console'
        try:
            print(f"{message.author.name}: {message.content}")
            response = self.send_message(f"{message.author.name}: {message.content}")
            print(f"IA: {response}")
            
            # Activar visualización de imagen durante el audio
            audio_arrays, duration_seconds = self.generate_audio(response)
            self.audio_to_reproduce = (True, duration_seconds)
            self.reproduce_audio(audio_arrays)
            self.audio_to_reproduce = (False, -1)
            
        except:
            pass
        await super().event_message(message)

    @commands.command()
    async def changevoice(self, ctx: commands.Context):
        'Change the voice of the bot'
        matches = re.match("!changevoice (.+)", ctx.message.content)
        if matches:
            await ctx.reply(matches.groups()[0][::-1])
        else:
            await ctx.reply(f"{ctx.author.name}, personaje, tienes que enviar el mensaje que quieres ver al revés.")