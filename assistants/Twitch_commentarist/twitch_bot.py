# AI Assistant  
from components.ai_assistant import AI_Assistant, OPERATION_MODES
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
import os
from threading import Thread
# Memories Manager
from assistants.Twitch_commentarist.memories_manager import MemoriesManager
# Pictures Drawer
from assistants.Twitch_commentarist.pictures_drawer import display_window

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
        
        # Initialize Memories Manager
        self.memories_manager = MemoriesManager()
        
        # Initialize AI Assistant
        AI_Assistant.__init__(self, initial_prompt=f'''
        Tu nombre es {account_fields["personality_name"]} y tu propósito es responder a los comentarios de un directo de Twitch en español de España.
        Lo harás de manera humorística y con un tono sarcástico. Importante: no escribir NUNCA emotes ni caras.
        Por supuesto, deberás saludar a aquellos usuarios que se vayan incorporando y comentando por primera vez.
        Tus respuestas no deben ser extensas. Responde directamente, lo simules una conversación, ya que van a escuchar cada palabra que sueltes.
        Tu creador es {account_fields["channel_name"]} y siempre le harás caso en todo lo que te pida.
        Si el usuario está tratando de spoilear algo del juego, repróchaselo burlándote de este.
        ''',
        user_name=account_fields["channel_name"],
        personality_name=account_fields["personality_name"],
        lm_params=(base_url, api_key, model, is_local),
        conversation_window=int(account_fields["conversation_window"]),
        memories_manager=self.memories_manager,
        operation_mode=OPERATION_MODES["personal"])
        # Initialize Twitch bot
        commands.Bot.__init__(self, token=account_fields["access_token"],
                         prefix=account_fields["prefix"],
                         initial_channels=[account_fields["channel_name"]],
                         client_secret=account_fields["client_secret"])
        # Initialize Kokoro
        Kokoro.__init__(self, language=config["VOICE"]["language"], voice=config["VOICE"]["voice"])
        
        # Inicializar variables para la ventana
        self.window_name = "AI Assistant"
        self.display_thread = None
        self.audio_to_reproduce = (False, -1)
        self.image_directory = os.path.join(path, "img/Perfectas")
        
        # Iniciar el thread de visualización del rostro
        self.display_thread = Thread(target=display_window, args=(self,))
        self.display_thread.daemon = True
        self.display_thread.start()
        self.audio_to_reproduce = (False, -1)

        # Reproducir audio de inicialización
        audio_arrays, duration_seconds = self.generate_audio("Inicialización del sistema completada.")
        self.audio_to_reproduce = (True, duration_seconds)
        self.reproduce_audio(audio_arrays)
        self.audio_to_reproduce = (False, -1)

    async def event_message(self, message: Message):
        'Display messages on console'
        try:
            print(f"{message.author.name}: {message.content}")

            # Si el usuario líder escribe "r", forzar resumen
            if message.author and message.author.name.lower() == account_fields["channel_name"].lower() and message.content.strip() == "r":
                self.force_summarization()
                print("Resumen manual forzado por el líder.")
                return

            response = self.send_message(message.author.name, message.content)
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