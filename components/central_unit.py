import json
import threading
from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Optional
import time

from components.ai_assistant import AI_Assistant
from components.transcriplator.transcriplator import Transcriplator
from components.kokoro.kokoro_class import Kokoro

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover
    sr = None


class InteractionMode(Enum):
    TEXT = 0
    VOICE = 1


class ActionType(Enum):
    """Acciones disponibles para el asistente."""
    CONTINUE = "Seguir hablando, sin ejecutar nada"
    TERMINATE = "Terminar conversación tras despedirte del usuario"


@dataclass
class LanguageModelSettings:
    base_url: str
    api_key: Optional[str]
    model: str
    is_local: bool


@dataclass
class AssistantSettings:
    initial_prompt: str
    personalities_path: str
    personality_name: str
    summarization_frequency: int
    auto_save: bool


@dataclass
class WakeSettings:
    phrase: str
    language: str
    energy_threshold: int
    pause_threshold: float
    dynamic_energy: bool
    ambient_duration: float
    phrase_time_limit: int


@dataclass
class VoiceSettings:
    language: str
    voice: str


class ConfigLoadError(RuntimeError):
    pass


class CentralUnitConfig:
    DEFAULT_PROMPT = (
        "Eres la unidad central de un ordenador personal llamada Her. "
        "Tu responsabilidad es conversar con el usuario, comprender su intención y "
        "decidir cuál de las acciones disponibles ejecutar. "
        "Respondes siempre en español."
    )

    def __init__(self, config_path: Path):
        config_path = Path(config_path)
        parser = ConfigParser(inline_comment_prefixes=("#", ";"))
        if not parser.read(config_path, encoding="utf-8"):
            raise ConfigLoadError(f"No se pudo leer el archivo de configuración en {config_path}")

        required_sections = ["LM", "CENTRAL_UNIT_CONFIG", "VOICE"]
        for section in required_sections:
            if not parser.has_section(section):
                raise ConfigLoadError(f"El archivo de configuración no define la sección [{section}]")

        self._config_path = config_path
        self._parser = parser
        self._central_section = parser["CENTRAL_UNIT_CONFIG"]
        self.lm = self._load_lm_settings()
        self.assistant = self._load_assistant_settings()
        self.voice = self._load_voice_settings()
        self.mode, self.wake_settings, self.transcriplator_config_path = self._load_central_settings()
        self.transcriplator_translate_override: Optional[bool] = None

    def _load_lm_settings(self) -> LanguageModelSettings:
        lm_section = self._parser["LM"]
        base_url = lm_section.get("base_url", fallback="").strip()
        model = lm_section.get("model", fallback="").strip()

        if not base_url:
            raise ConfigLoadError("base_url es obligatorio en la sección [LM]")
        if not model:
            raise ConfigLoadError("model es obligatorio en la sección [LM]")

        api_key = lm_section.get("api_key", fallback=None)
        api_key = api_key.strip() if api_key else None

        return LanguageModelSettings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            is_local=not api_key,
        )

    def _load_assistant_settings(self) -> AssistantSettings:
        section = self._central_section
        personalities_path = section.get("personalities_path", fallback=None)
        personality_name = section.get("personality_name", fallback=None)

        # Fallback a configuración de Twitch si existe
        if (not personalities_path or not personality_name) and self._parser.has_section("TWITCH_COMMENTARIST_CONFIG"):
            twitch_section = self._parser["TWITCH_COMMENTARIST_CONFIG"]
            personalities_path = personalities_path or twitch_section.get("personalities_path", fallback=str(Path.home() / "Her"))
            personality_name = personality_name or twitch_section.get("personality_name", fallback="default")

        personalities_path = personalities_path or str(Path.home() / "Her")
        personality_name = personality_name or "default"

        return AssistantSettings(
            initial_prompt=section.get("initial_prompt", fallback=self.DEFAULT_PROMPT).strip(),
            personalities_path=personalities_path,
            personality_name=personality_name,
            summarization_frequency=section.getint("summarization_frequency", fallback=10),
            auto_save=section.getboolean("auto_save", fallback=True),
        )

    def _load_voice_settings(self) -> VoiceSettings:
        voice_section = self._parser["VOICE"]
        language = voice_section.get("language", fallback="spanish").strip()
        voice = voice_section.get("voice", fallback="hf_beta").strip()

        if not language or not voice:
            raise ConfigLoadError("Los campos language y voice son obligatorios en la sección [VOICE]")

        return VoiceSettings(language=language, voice=voice)

    def _load_central_settings(self):
        try:
            mode = InteractionMode(self._central_section.getint("mode", fallback=0))
        except ValueError as exc:
            raise ConfigLoadError("El valor de mode en [CENTRAL_UNIT_CONFIG] no es válido") from exc

        wake_settings = WakeSettings(
            phrase=self._central_section.get("wake_phrase", fallback="ORDENADOR AL HABLA").strip(),
            language=self._central_section.get("wake_language", fallback="es-ES").strip(),
            energy_threshold=self._central_section.getint("wake_energy_threshold", fallback=300),
            pause_threshold=self._central_section.getfloat("wake_pause_threshold", fallback=0.8),
            dynamic_energy=self._central_section.getboolean("wake_dynamic_energy", fallback=True),
            ambient_duration=self._central_section.getfloat("wake_ambient_duration", fallback=1.5),
            phrase_time_limit=self._central_section.getint("wake_phrase_time_limit", fallback=3),
        )

        transcriplator_config = self._central_section.get(
            "transcriplator_config",
            fallback="components/transcriplator/config.ini",
        ).strip() or "components/transcriplator/config.ini"
        
        transcriplator_config_path = (self._config_path.parent / transcriplator_config).resolve()

        return mode, wake_settings, transcriplator_config_path


class WakeWordListener:
    def __init__(self, settings: WakeSettings):
        if sr is None:
            raise RuntimeError(
                "El módulo speech_recognition no está disponible. Instálalo para habilitar el modo de activación por voz."
            )
        self.settings = settings
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = settings.energy_threshold
        self._recognizer.dynamic_energy_threshold = settings.dynamic_energy
        self._recognizer.pause_threshold = settings.pause_threshold

    def wait_for_activation(self):
        microphone = sr.Microphone()
        with microphone as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=self.settings.ambient_duration)
            while True:
                try:
                    audio = self._recognizer.listen(source, phrase_time_limit=self.settings.phrase_time_limit)
                except sr.WaitTimeoutError:
                    continue

                try:
                    transcript = self._recognizer.recognize_google(audio, language=self.settings.language)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as exc:
                    raise RuntimeError("No fue posible conectarse con la API de reconocimiento de voz de Google") from exc

                # Normalizar ambos textos para comparación
                normalized_transcript = self._normalize(transcript)
                normalized_phrase = self._normalize(self.settings.phrase)
                
                print(f"Transcrito: {normalized_transcript}")
                print(f"Frase de activación: {normalized_phrase}")
                
                if normalized_transcript == normalized_phrase:
                    return

    @staticmethod
    def _normalize(text: str) -> str:
        """Normaliza texto eliminando puntuación y convirtiendo a minúsculas."""
        normalized = text.strip().lower()
        for char in ["?", "¿", "¡", "."]:
            normalized = normalized.replace(char, "")
        return normalized


class ActionParser:
    def parse(self, assistant_output: Optional[str]):
        if not assistant_output or not assistant_output.strip():
            return ActionType.CONTINUE, ""

        cleaned = assistant_output.strip()
        json_text = self._extract_json(cleaned)
        
        if not json_text:
            return ActionType.CONTINUE, cleaned

        try:
            payload = json.loads(json_text, strict=False)
        except json.JSONDecodeError:
            return ActionType.CONTINUE, cleaned

        action_value = str(payload.get("action", "")).strip().upper()
        message = str(payload.get("message", "")).strip()

        if action_value == "TERMINATE":
            return ActionType.TERMINATE, message
        if action_value == "CONTINUE":
            return ActionType.CONTINUE, message

        return ActionType.CONTINUE, message or cleaned

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None
        return text[start : end + 1]


class TranscriplatorVoiceSession:
    def __init__(self, config_path: Path, translate_to_english: Optional[bool] = None, speech_start_callback=None):
        self._queue: Queue = Queue()
        self._transcriplator = Transcriplator(
            config_file=str(config_path),
            transcription_callback=self._queue.put,
            enable_gui=False,
            use_keyboard=False,
            speech_start_callback=speech_start_callback,
            translate_to_english=translate_to_english,
        )

        if not self._transcriplator.initialize():
            raise RuntimeError("No fue posible inicializar el componente de transcripción (Transcriplator).")

        if not self._transcriplator.recorder:
            raise RuntimeError("La inicialización del Transcriplator no creó un grabador de audio válido.")

        self._thread = threading.Thread(target=self._transcriplator.recorder.run, daemon=True)
        self._thread.start()

    def get_message(self) -> str:
        return self._queue.get()

    def stop(self):
        self._transcriplator.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class CentralUnit:
    def __init__(self, config: CentralUnitConfig, translate_to_english: Optional[bool] = False):
        self.config = config
        self.action_parser = ActionParser()

        # Construir prompt enriquecido con instrucciones JSON
        enriched_prompt = self._build_enriched_prompt(config.assistant.initial_prompt)

        self.assistant = AI_Assistant(
            initial_prompt=enriched_prompt,
            personalities_path=config.assistant.personalities_path,
            personality_name=config.assistant.personality_name,
            summarization_frequency=config.assistant.summarization_frequency,
            auto_save=config.assistant.auto_save,
            lm_params=(config.lm.base_url, config.lm.api_key, config.lm.model, config.lm.is_local),
        )

        self.wake_listener = WakeWordListener(config.wake_settings)
        self.mode = config.mode
        self.transcriplator_config_path = config.transcriplator_config_path
        self.transcriplator_translate_override: Optional[bool] = translate_to_english
        
        try:
            self.voice_synth = Kokoro(config.voice.language, config.voice.voice)
        except Exception as exc:
            raise ConfigLoadError(f"No se pudo inicializar Kokoro: {exc}") from exc

    @staticmethod
    def _build_enriched_prompt(base_prompt: str) -> str:
        """Construye el prompt con instrucciones de formato JSON y estilo conversacional."""
        # Generar lista de acciones dinámicamente desde el enum
        action_names = "|".join(action.name for action in ActionType)
        actions_list = "\n".join(f"- {action.name}: {action.value}" for action in ActionType)
        
        return (
            f"{base_prompt}\n\n"
            f'Debes responder SIEMPRE en formato JSON válido con la forma '
            f'{{"action":"<{action_names}>","message":"<tu respuesta al usuario>"}}.\n'
            f"Las acciones disponibles son:\n{actions_list}\n"
            "La conversación es oral: el campo message debe sonar natural y cercano, como hablando con un amigo.\n"
            "Evita enumeraciones numéricas o con viñetas y cualquier formato rígido; expresa la información con frases fluidas.\n"
            "No repitas palabras innecesarias ni utilices caracteres especiales (p. ej. paréntesis o corchetes) para presentar opciones.\n"
            "Trata ser breve y concisa, extendiéndote únicamente si el usuario te pide que lo hagas."
            "El campo message debe contener la respuesta que el usuario escuchará, en español."
        )

    @classmethod
    def from_file(cls, config_path: Path, translate_to_english: Optional[bool] = False):
        config = CentralUnitConfig(config_path)
        return cls(config, translate_to_english=translate_to_english)

    def run(self):
        while True:
            try:
                self._wait_for_activation()
                self._conversation_loop()
            except KeyboardInterrupt:
                break

    def _wait_for_activation(self):
        print("Sistema en espera. Pronuncia la frase de activación para comenzar.")
        self.wake_listener.wait_for_activation()
        print("Unidad central activada. Puedes comenzar a hablar.")

    def _conversation_loop(self):
        voice_session = None
        try:
            if self.mode is InteractionMode.VOICE:
                voice_session = self._start_voice_session()

            # Saludo inicial
            greeting_action = self._send_greeting()
            if greeting_action is ActionType.TERMINATE:
                return

            # Loop principal de conversación
            while True:
                user_message = self._get_user_input(voice_session)
                if not user_message:
                    continue

                start_time = time.time()
                action, response = self._process_message(user_message)
                elapsed = time.time() - start_time

                if response:
                    self._deliver_response(response, elapsed)

                if action is ActionType.TERMINATE:
                    break
        finally:
            if voice_session:
                voice_session.stop()

    def _send_greeting(self):
        """Envía el saludo inicial tras la activación."""
        instruction = (
            "La conversación acaba de comenzar tras la frase de activación. "
            "Saluda al usuario con un mensaje breve y cordial. "
            "Asegúrate de responder con action CONTINUE para seguir hablando."
        )
        start_time = time.time()
        action, response = self._process_message(instruction)
        elapsed = time.time() - start_time
        
        if response:
            self._deliver_response(response, elapsed)
        return action

    def _get_user_input(self, voice_session: Optional[TranscriplatorVoiceSession]):
        """Obtiene input del usuario (texto o voz según el modo)."""
        if self.mode is InteractionMode.TEXT:
            user_input = input("Usuario: ").strip()
            return user_input or None

        if voice_session is None:
            raise RuntimeError("La sesión de voz no está inicializada.")

        transcript = voice_session.get_message()
        if transcript:
            print(f"Usuario (voz): {transcript}")
        return transcript

    def _process_message(self, user_message: str):
        """Procesa un mensaje del usuario y retorna la acción y respuesta."""
        assistant_output = self.assistant.send_message(user_message)
        return self.action_parser.parse(assistant_output)

    def _start_voice_session(self) -> TranscriplatorVoiceSession:
        print("Inicializando módulo de transcripción por voz...")
        try:
            return TranscriplatorVoiceSession(
                self.transcriplator_config_path,
                translate_to_english=self.transcriplator_translate_override,
                speech_start_callback=self._on_user_speech_start,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"No se pudo iniciar el transcriptor de voz: {exc}") from exc

    def _speak(self, text: str):
        """Sintetiza y reproduce voz."""
        if not text:
            return
        try:
            self.voice_synth.stop_playback()
            audio_arrays, _ = self.voice_synth.generate_audio(text)
            self.voice_synth.reproduce_audio(audio_arrays)
        except Exception as exc:
            print(f"Aviso: no se pudo sintetizar voz ({exc})")

    def _on_user_speech_start(self):
        """Callback cuando el usuario empieza a hablar (detiene reproducción actual)."""
        if self.mode is InteractionMode.VOICE:
            try:
                self.voice_synth.stop_playback()
            except Exception:
                pass

    def _deliver_response(self, message: str, elapsed: Optional[float] = None):
        """Entrega la respuesta del asistente al usuario."""
        if not message:
            return
        
        print(f"Her: {message}")
        if elapsed is not None:
            print(f"Tiempo de respuesta: {elapsed:.3f}s")
        
        if self.mode is InteractionMode.VOICE:
            self._speak(message)
