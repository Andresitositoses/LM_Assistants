from openai import OpenAI
import os
import platform

class AI_Assistant():

    FIELDS_SEPARATOR = "|/="

    def __init__(self, initial_prompt: str, personalities_path: str, personality_name: str, summarization_frequency: int, auto_save: bool, lm_params: tuple, include_ai_in_history: bool = False, memories_manager=None):
        '''
        initial_prompt: str -> Prompt inicial para el asistente
        personalities_path: str -> Ruta a la carpeta de personalidades
        personality_name: str -> Nombre de la personalidad del asistente
        summarization_frequency: int -> Frecuencia de resumen (en conversaciones). < 0 -> No se realiza ningún resumen.
        auto_save: bool -> Si se debe guardar el estado del asistente en un archivo de texto.
        lm_params: tuple -> (base_url, api_key, model, is_local) - Parámetros del modelo de lenguaje
        include_ai_in_history: bool -> Si se deben incluir las respuestas de la IA en el historial de conversación (por defecto False)
        memories_manager: MemoriesManager -> Instancia de MemoriesManager para gestionar personalidades en MongoDB (opcional)
        '''

        # Extraer parámetros de la tupla
        base_url, api_key, model, is_local = lm_params
        
        # Validar parámetros LM requeridos
        err = self.validate_LM_params(base_url, api_key, model, is_local)
        if err is not None:
            raise ValueError(err)
        
        # Mostrar qué tipo de modelo se está usando
        model_type = "local" if is_local else "en la nube"
        print(f"Usando modelo {model_type}: {model}")
        
        # Para modelos locales, la API key no es requerida
        if is_local:
            api_key = api_key or "not-required"
        
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Parameters
        self.initial_prompt = initial_prompt
        self.personality_name = personality_name
        self.summarization_frequency = summarization_frequency
        self.summarization_counter = 0
        self.auto_save = auto_save
        self.include_ai_in_history = include_ai_in_history
        self.memories_manager = memories_manager
        
        # Configurar el directorio de personalidades
        self.personalities_path = personalities_path

        # Conversation history
        self.conversation_history = []
        if self.has_status():
            self.load_status()
        # No añadimos el initial_prompt aquí porque send_message lo añade 
        # dinámicamente al principio de cada consulta al modelo.
        
    def validate_LM_params(self, base_url, api_key, model, is_local):
        """Valida los parámetros de configuración del modelo de lenguaje"""
        if not base_url:
            return "base_url es requerido"
        if not model:
            return "model es requerido"
        if not is_local and not api_key:
            return "api_key es requerido para modelos en la nube"
        return None

    def send_message(self, message):
        '''
        Add a message to the conversation history and its response.
        Returns the last message from the assistant.
        '''

        self.conversation_history.append({"role": "user", "content": message})
        try:
            # Preparar mensajes para el modelo combinando initial_prompt con el historial
            messages_to_send = []
            
            # Si el primer mensaje es el resumen, combinarlo con initial_prompt
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                system_content = self.conversation_history[0]["content"]
                # Combinar initial_prompt con el resumen almacenado
                combined_content = f"{self.initial_prompt}. {system_content}"
                messages_to_send.append({"role": "system", "content": combined_content})
                # Agregar el resto del historial
                messages_to_send.extend(self.conversation_history[1:])
            else:
                # Si no hay resumen, usar solo initial_prompt
                messages_to_send.append({"role": "system", "content": self.initial_prompt})
                messages_to_send.extend(self.conversation_history)
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            ai_response = completion.choices[0].message.content

            if self.include_ai_in_history:
                self.conversation_history.append({"role": "assistant", "content": ai_response})

            if self.summarization_frequency > 0 and self.summarization_counter >= self.summarization_frequency:
                print("Performing summarization...")
                self.perform_summarization(self.auto_save)
                self.summarization_counter = 0
            else:
                self.summarization_counter += 1

            return ai_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
        
    def perform_summarization(self, save: bool):
        try:
            previous_summary = ""
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                parts = self.conversation_history[0]['content'].split(self.FIELDS_SEPARATOR)
                if len(parts) >= 2:
                    previous_summary = parts[1]

            text = "Haz un resumen de los aspectos más relevantes de la conversación actual. " \
                   "IMPORTANTE: Debes combinar este nuevo conocimiento con el resumen anterior (si existe), " \
                   "asegurándote de MANTENER todos los puntos y recuerdos almacenados previamente, a menos que sean redundantes. " \
                   "No debes olvidar los puntos importantes ya memorizados."
            
            if previous_summary:
                text += f"\n\nResumen Anterior (NO OLVIDAR): {previous_summary}"
            self.conversation_history.append({"role": "user", "content": text})

            # Preparar mensajes combinando initial_prompt con el historial
            messages_to_send = []
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                system_content = self.conversation_history[0]["content"]
                combined_content = f"{self.initial_prompt}. {system_content}"
                messages_to_send.append({"role": "system", "content": combined_content})
                messages_to_send.extend(self.conversation_history[1:])
            else:
                messages_to_send.append({"role": "system", "content": self.initial_prompt})
                messages_to_send.extend(self.conversation_history)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            ai_response = completion.choices[0].message.content

            self.conversation_history.clear()
            # Solo guardamos el resumen, no el initial_prompt (ya lo tenemos como parámetro)
            self.conversation_history.append({"role": "system", "content": f"Aprendido en conversaciones anteriores:{self.FIELDS_SEPARATOR} {ai_response}"})
            print(f"\n\n(System) Resumen guardado: {ai_response}\n\n")

            if save:
                self.save_status()

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def force_summarization(self):
        '''
        Forzar un resumen manual de la conversación.
        '''
        print("Forzando resumen manual...")
        self.perform_summarization(self.auto_save)
        self.summarization_counter = 0
        
    def load_status(self):
        """Carga la personalidad desde MongoDB"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        try:
            content = self.memories_manager.load_personality(self.personality_name)
            if content:
                # Dividir por el separador principal
                parts = content.split(self.FIELDS_SEPARATOR)
                if len(parts) >= 2:
                    # El primer elemento es el rol (system)
                    role = parts[0]
                    # El segundo elemento es el contenido
                    content_text = parts[1]
                    # Si hay más partes, son parte del contenido
                    if len(parts) > 2:
                        content_text += self.FIELDS_SEPARATOR + parts[2]
                    self.conversation_history.append({"role": role, "content": content_text})
                    print(f"Personalidad cargada desde MongoDB: {self.personality_name}")
        except Exception as e:
            print(f"Error al cargar desde MongoDB: {str(e)}")
        
    def save_status(self):
        """Guarda la personalidad en MongoDB"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        # Preparar el contenido a guardar
        content = ""
        for item in self.conversation_history:
            content += f"{item['role']}{self.FIELDS_SEPARATOR}{item['content']}\n"
        
        # Guardar en MongoDB
        try:
            self.memories_manager.save_personality(self.personality_name, content)
            print(f"Personalidad guardada en MongoDB: {self.personality_name}")
        except Exception as e:
            print(f"Error al guardar en MongoDB: {str(e)}")

    def has_status(self) -> bool:
        """Verifica si existe la personalidad en MongoDB"""
        if not self.memories_manager:
            return False
            
        try:
            return self.memories_manager.has_personality(self.personality_name)
        except Exception as e:
            print(f"Error al verificar en MongoDB: {str(e)}")
            return False