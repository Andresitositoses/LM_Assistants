from openai import OpenAI
import time

OPERATION_MODES = {
    "personal": 0, # Unique user
    "group": 1 # Several users
}

class AssistantPersonality:
    def __init__(self, name: str, summary: str, latest_messages_history: list, time_to_sleep: int):
        self.name = name
        self.summary = summary
        self.latest_messages_history = latest_messages_history
        self.time_to_sleep = time_to_sleep # Remaining iterations before performing a summarization

class UserMemory:
    def __init__(self, name: str, summary: str, latest_messages_history: list, time_to_sleep: int):
        self.name = name
        self.summary = summary
        self.latest_messages_history = latest_messages_history
        self.time_to_sleep = time_to_sleep

class AI_Assistant():

    def __init__(self, initial_prompt: str, user_name:str, personality_name:str, lm_params: tuple, conversation_window = 10, memories_manager = None, operation_mode = OPERATION_MODES["personal"]):
        '''
        initial_prompt: str -> Prompt inicial para el asistente
        personality_name: str -> Nombre de la personalidad del asistente
        lm_params: tuple -> (base_url, api_key, model, is_local) - Parámetros del modelo de lenguaje
        conversation_window: int -> Número de mensajes a considerar en la conversación
        memories_manager: MemoriesManager -> Instancia de MemoriesManager para gestionar personalidades (opcional)
        '''

        # Extraer parámetros de la tupla
        base_url, api_key, model, is_local = lm_params
        
        # Validar parámetros LM requeridos
        err = self.validate_LM_params(base_url, api_key, model, is_local)
        if err is not None:
            raise ValueError(err)

        print(f"Usando modelo {'local' if is_local else 'en la nube'}: {model}")
        
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=(api_key or "not-required") if is_local else api_key,
        )

        # Parameters
        self.initial_prompt = initial_prompt
        self.user_name = user_name
        self.personality_name = personality_name
        self.conversation_window = conversation_window
        self.conversation_history = [] # Global conversation history
        self.memories_manager = memories_manager
        self.assistant_personality: AssistantPersonality = None
        self.operation_mode = operation_mode

        # Initialize 
        if self.is_existing_personality():
            self.load_personality()
        else:
            self.create_personality()
        
    def validate_LM_params(self, base_url, api_key, model, is_local):
        """Valida los parámetros de configuración del modelo de lenguaje"""
        if not base_url:
            return "base_url es requerido"
        if not model:
            return "model es requerido"
        if not is_local and not api_key:
            return "api_key es requerido para modelos en la nube"
        return None
        
    def send_message(self, message_author, message_content):
        '''
        Add a message to the conversation history and its response.
        Returns the last message from the assistant.
        '''

        self.conversation_history.append({"role": "user", "content": f"{message_author}: {message_content}", "timestamp": time.time()})
        try:
            # Preparar mensajes para el modelo combinando lo siguiente:
            # - initial_prompt (system)
            # - resumen del asistente (system)
            # - resumen e historial del usuario (si procede) (system)
            # - historial de conversación (user and assistant)
            # - nuevo mensaje (user)
            messages_to_send = []
            messages_to_send.append({"role": "system", "content": self.initial_prompt}) # Initial prompt
            messages_to_send.append({"role": "system", "content": self.assistant_personality.summary}) # Assistant summary
            combined_history = []
            if self.operation_mode == OPERATION_MODES["group"] and message_author != self.user_name:
                is_new_user = False
                if not self.is_existing_user(message_author): # If the memory doesn't exist, create it
                    self.create_user_memory(message_author)
                    is_new_user = True
                user_memory = self.load_user_memory(message_author)
                # User summary and history (both global chat and user memory)
                if is_new_user:
                    messages_to_send.append({"role": "system", "content": f"El usuario '{message_author}' se ha unido al chat por primera vez.\n\n"})
                else:
                    messages_to_send.append({"role": "system", "content": f"Resumen del usuario {message_author}: {user_memory.summary}\n\n"})
                combined_history = sorted(
                    self.conversation_history[:-1] + (user_memory.latest_messages_history if user_memory else []),
                    key=lambda x: x.get('timestamp', 0)
                )
                messages_to_send.extend([{'role': m['role'], 'content': m['content']} for m in combined_history])
            else:
                # Combinar historial global con el historial de la personalidad
                combined_history = sorted(
                    self.conversation_history[:-1] + (self.assistant_personality.latest_messages_history if self.assistant_personality else []),
                    key=lambda x: x.get('timestamp', 0)
                )
                messages_to_send.extend([{'role': m['role'], 'content': m['content']} for m in combined_history])
            messages_to_send.append({"role": "user", "content": f"{message_author}: {message_content}"}) # New message
            
            # Specify the format of the response
            request_format_specification = """
            Analiza la solicitud y decide si se debe responder al último mensaje o no, basándote en los siguientes criterios:
            
            Criterios para responder (hay_respuesta: True):
            1. El mensaje no es spam, un comando o un emote.
            2. El mensaje es relevante al contexto del chat y se va a aportar algún comentario gracioso.
            3. El mensaje es en respuesta a otro usuario y hay algo que aportar.

            Formato de respuesta obligatorio:
            *Ejemplo: True/False&&&Contenido de la respuesta/Texto 'No cumple con el criterio N'
            """
            messages_to_send.append({"role": "system", "content": request_format_specification})

            # Send messages to the model
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            ai_response = completion.choices[0].message.content
            is_there_response = False
            if ai_response and "&&&" in ai_response:
                is_there_response = ai_response.split("&&&", 1)[0].strip() == "True"
                ai_response = ai_response.split("&&&", 1)[1].strip()
            else:
                return False, None

            if ai_response and ":" in ai_response:
                ai_response = ai_response.split(":", 1)[1].strip()

            # Add assistant response to conversation history
            if is_there_response:
                self.conversation_history.append({"role": "assistant", "content": f"{self.assistant_personality.name}: {ai_response}", "timestamp": time.time()})

            # Update messages histories
            if self.operation_mode == OPERATION_MODES["group"]:
                # Actualizamos el historial de mensajes de la personalidad (la IA)
                if message_author == self.user_name:
                    # Agregar el nuevo mensaje al historial de la personalidad (basado en el asistente y el usuario principal)
                    self.assistant_personality.latest_messages_history.append({"role": "user", "content": f"{message_author}: {message_content}", "timestamp": time.time()})
                    if is_there_response:
                        self.assistant_personality.latest_messages_history.append({"role": "assistant", "content": f"{self.assistant_personality.name}: {ai_response}", "timestamp": time.time()})
                else:
                    # Agregar los nuevos mensajes al historial del usuario
                    user_memory = self.load_user_memory(message_author)
                    user_memory.latest_messages_history.append({"role": "user", "content": f"{message_author}: {message_content}", "timestamp": time.time()})
                    if is_there_response:
                        user_memory.latest_messages_history.append({"role": "assistant", "content": f"{self.assistant_personality.name}: {ai_response}", "timestamp": time.time()})
                    self.update_user_memory(user_memory)
            else:
                if message_author == self.user_name:
                    # Agregar el nuevo mensaje al historial de la personalidad (basado en el asistente y el usuario principal)
                    self.assistant_personality.latest_messages_history.append({"role": "user", "content": f"{message_author}: {message_content}", "timestamp": time.time()})
                    if is_there_response:
                        self.assistant_personality.latest_messages_history.append({"role": "assistant", "content": f"{self.assistant_personality.name}: {ai_response}", "timestamp": time.time()})

            # Comprobar TTS del asistente, realizar resumen si es necesario y actualizar el estado del asistente
            if is_there_response:
                self.assistant_personality.time_to_sleep -= 1
                if self.is_time_to_summarize():
                    self.perform_assistant_summarization()
            self.update_personality()

            # Si procede, actualizar la memoria del usuario
            if self.operation_mode == OPERATION_MODES["group"] and message_author != self.user_name:
                user_memory = self.load_user_memory(message_author)
                user_memory.time_to_sleep -= 1
                if self.user_has_to_summarize(user_memory):
                    self.perform_user_memory_summarization(user_memory, combined_history)
                self.update_user_memory(user_memory)

            # Limpiar por la cola el historial de conversación (Dejamos solo los últimos <conversation_window> mensajes)
            self.conversation_history = self.conversation_history[-self.conversation_window:]

            # Suprimimos el autor del mensaje
            return is_there_response, ai_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False, None

    def user_has_to_summarize(self, user: UserMemory):
        return user.time_to_sleep == 0

    def perform_user_memory_summarization(self, user: UserMemory, combined_history: list):
        try:
            summarization_request = f'''
            Hazte un resumen de los aspectos más relevantes de la conversación actual con este usuario, hablándote a ti misma. 
            Asegúrate de MANTENER todos los puntos y recuerdos almacenados previamente, a menos que sean redundantes o irrelevantes.
            '''
            summarization_request += f"IMPORTANTE: Debes combinar este nuevo conocimiento con el resumen anterior: {user.summary}\n\n" if user.summary else "\n\n"
            summarization_request += f"Conversación:\n{"\n".join(msg['content'] for msg in combined_history)}\n\n"
            
            # Preparar mensajes combinando la identidad del asistente, la del usuario, el historial de mensaje (chat y usuario) y la petición de resumen
            messages_to_send = []
            messages_to_send.append({"role": "system", "content": self.initial_prompt}) # Initial prompt
            messages_to_send.append({"role": "system", "content": self.assistant_personality.summary}) # Assistant summary
            messages_to_send.append({"role": "user", "content": summarization_request}) # Summarization request

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            user.summary = completion.choices[0].message.content
            user.latest_messages_history = []
            user.time_to_sleep = self.conversation_window
        except Exception as e:
            print(f"Error: {str(e)}")

    def load_user_memory(self, user_name: str):
        "Carga la memoria del usuario"
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        try:
            user_memory = self.memories_manager.load_memory(user_name)
            if user_memory:
                return UserMemory(user_name, user_memory["summary"], user_memory["latest_messages_history"], user_memory["time_to_sleep"])
            else:
                return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def update_user_memory(self, user_memory: UserMemory):
        """Guarda la memoria del usuario"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        try:
            self.memories_manager.save_memory(user_memory.name, user_memory.summary, user_memory.latest_messages_history, user_memory.time_to_sleep)
            print(f"Memoria guardada: {user_memory.name}")
        except Exception as e:
            print(f"Error al guardar: {str(e)}")

    def is_existing_user(self, user_name: str):
        '''
        Verifica si existe la memoria del usuario
        '''
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return False
        
        try:
            return self.memories_manager.has_memory(user_name)
        except Exception as e:
            print(f"Error al verificar: {str(e)}")
            return False

    def create_user_memory(self, user_name: str):
        '''
        Crea la memoria del usuario
        '''
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
        
        try:
            self.memories_manager.create_memory(user_name, self.conversation_window)
            print(f"Memoria creada: {user_name}")
        except Exception as e:
            print(f"Error al crear: {str(e)}")

    def is_time_to_summarize(self):
        return self.assistant_personality.time_to_sleep == 0
        
    def perform_assistant_summarization(self):
        try:    
            summarization_request = f'''
            Hazte un resumen de los aspectos más relevantes de la conversación actual, hablándote a ti misma. 
            Asegúrate de MANTENER todos los puntos y recuerdos almacenados previamente, a menos que sean redundantes o irrelevantes.
            '''
            summarization_request += f"IMPORTANTE: Debes combinar este nuevo conocimiento con el resumen anterior: {self.assistant_personality.summary}\n\n" if self.assistant_personality.summary else "\n\n"
            summarization_request += f"Conversación:\n{"\n".join(msg['content'] for msg in self.assistant_personality.latest_messages_history)}\n\n"
            
            # Preparar mensajes combinando initial_prompt con el historial
            messages_to_send = []
            messages_to_send.append({"role": "system", "content": self.initial_prompt}) # Initial prompt
            messages_to_send.append({"role": "system", "content": self.assistant_personality.summary}) # Assistant summary
            messages_to_send.append({"role": "user", "content": summarization_request}) # Summarization request

            # Send messages to the model
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            ai_response = completion.choices[0].message.content

            # Guardamos el resumen
            self.assistant_personality.summary = ai_response
            self.assistant_personality.latest_messages_history = []
            self.assistant_personality.time_to_sleep = self.conversation_window
            print(f"\n\n(System) Resumen guardado: {ai_response}\n\n")

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def force_summarization(self):
        '''
        Forzar un resumen manual de la conversación.
        '''
        print("Forzando resumen manual...")
        self.perform_assistant_summarization()
        self.update_personality()
        
    def load_personality(self):
        """Carga la personalidad"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        try:
            personality = self.memories_manager.load_personality(self.personality_name)
            if personality:
                # Cargar información previa del asistente
                self.assistant_personality = AssistantPersonality(self.personality_name, personality["summary"], personality["latest_messages_history"], personality["time_to_sleep"])
                print(f"Personalidad cargada: {self.personality_name}")
        except Exception as e:
            print(f"Error al cargar: {str(e)}")
        
    def update_personality(self):
        """Guarda la personalidad"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        try:
            self.memories_manager.save_personality(self.assistant_personality.name,
                                                    self.assistant_personality.latest_messages_history,
                                                    self.assistant_personality.summary,
                                                    self.assistant_personality.time_to_sleep)
            print(f"Personalidad guardada: {self.assistant_personality.name}")
        except Exception as e:
            print(f"Error al guardar: {str(e)}")

    def is_existing_personality(self) -> bool:
        """Verifica si existe la personalidad"""
        if not self.memories_manager:
            return False
            
        try:
            return self.memories_manager.has_personality(self.personality_name)
        except Exception as e:
            print(f"Error al verificar: {str(e)}")
            return False

    def create_personality(self):
        """Crea la personalidad"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        try:
            self.memories_manager.create_personality(self.personality_name, self.conversation_window)
            print(f"Personalidad creada: {self.personality_name}")
        except Exception as e:
            print(f"Error al crear: {str(e)}")