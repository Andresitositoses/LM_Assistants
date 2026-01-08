from openai import OpenAI
import os
import platform
from datetime import datetime

class AI_Assistant():

    FIELDS_SEPARATOR = "|/="

    def __init__(self, initial_prompt: str, personalities_path: str, personality_name: str, summarization_frequency: int, auto_save: bool, lm_params: tuple, include_ai_in_history: bool = False, memories_manager=None, mode: str = "personal", leader_name: str = None):
        '''
        initial_prompt: str -> Prompt inicial para el asistente
        personalities_path: str -> Ruta a la carpeta de personalidades
        personality_name: str -> Nombre de la personalidad del asistente
        summarization_frequency: int -> Frecuencia de resumen (en conversaciones). < 0 -> No se realiza ningún resumen.
        auto_save: bool -> Si se debe guardar el estado del asistente en un archivo de texto.
        lm_params: tuple -> (base_url, api_key, model, is_local) - Parámetros del modelo de lenguaje
        include_ai_in_history: bool -> Si se deben incluir las respuestas de la IA en el historial de conversación (por defecto False)
        memories_manager: MemoriesManager -> Instancia de MemoriesManager para gestionar personalidades en MongoDB (opcional)
        mode: str -> 'personal' o 'chat'. Define el comportamiento del asistente.
        leader_name: str -> Nombre del usuario "líder" (requerido para mode='chat')
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
        self.mode = mode
        self.leader_name = leader_name
        
        # Stream buffer for chat mode
        self.stream_buffer = [] # Lista de últimos mensajes del stream
        
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

    def send_message(self, message, author=None, timestamp=None):
        '''
        Add a message to the conversation history and its response.
        Returns the last message from the assistant.
        '''
        if self.mode == "personal":
            content_to_store = f"{author}: {message}" if author else message
            self.conversation_history.append({"role": "user", "content": content_to_store})
        elif self.mode == "chat":
            # Logic for Leader
            if author.lower() == self.leader_name.lower():
                self.conversation_history.append({"role": "user", "content": f"{author}: {message}"})
            
            # Logic for non-leader users (User Memory)
            user_summary = ""
            user_messages = []
            if author.lower() != self.leader_name.lower():
                    user_summary, user_messages = self._update_chat_user_memory(author, message, author, timestamp)

        try:
            # Preparar mensajes para el modelo
            messages_to_send = []
            
            if self.mode == "personal":
                # Si el primer mensaje es el resumen, combinarlo con initial_prompt
                if self.conversation_history and self.conversation_history[0]["role"] == "system":
                    system_content = self.conversation_history[0]["content"]
                    combined_content = f"{self.initial_prompt}. {system_content}"
                    messages_to_send.append({"role": "system", "content": combined_content})
                    messages_to_send.extend(self.conversation_history[1:])
                else:
                    messages_to_send.append({"role": "system", "content": self.initial_prompt})
                    messages_to_send.extend(self.conversation_history)
            
            elif self.mode == "chat":
                # Construct Chat Mode Prompt
                # 1. System Prompt (invariant) 
                
                system_content = self.initial_prompt
                # Add Leader's "Persistent" Memory/Summary to the system prompt context if available
                if self.conversation_history and self.conversation_history[0]["role"] == "system":
                     system_content += f"\n{self.conversation_history[0]['content']}"

                messages_to_send.append({"role": "system", "content": system_content})

                #TODO: Hay que incluir el resumen de la propia personalidad

                # Add a welcome message if it's a new user
                if not user_summary and len(user_messages) <= 1 and author and author.lower() != self.leader_name.lower():
                     # New user scenario
                     messages_to_send.append({"role": "system", "content": f"El usuario {author} es nuevo. Dale la bienvenida."})
                
                # Add User's Memory/Summary to the system prompt context if available
                if user_summary:
                    messages_to_send.append({"role": "system", "content": f"Resumen sobre {author}: {user_summary}"})

                # Concatenación ordenada de stream y user messages
                # 1. Select History Source
                history_to_add = []
                if author and self.leader_name and author.lower() == self.leader_name.lower():
                    # For leader, use conversation_history
                    # Filter out system messages if they are not relevant or just take the chat flow
                    # Usually conversation_history[0] is system/prompt/summary.
                    if len(self.conversation_history) > 1:
                         history_to_add = self.conversation_history[1:] # Exclude initial system prompt
                else:
                    # For other users, use user_messages
                    if len(user_messages) > 1:
                         history_to_add = []
                         # Convert user_memory format to {"role", "content"}
                         for msg in user_messages[:-1]: # Exclude current
                             role = "user"
                             content = f"{msg['author']}: {msg['content']}"
                             if msg.get('author') == self.personality_name:
                                 role = "assistant"
                                 content = msg['content']
                             history_to_add.append({"role": role, "content": content})

                # 2. Add History to messages_to_send
                if history_to_add:
                     # Ensure we are adding valid role/content dicts
                     # If coming from conversation_history they are already dicts
                     # If coming from user_messages we just formatted them above
                     messages_to_send.extend(history_to_add)


            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            ai_response = completion.choices[0].message.content

            if self.include_ai_in_history:
                 # In chat mode, only append if it's the leader (or fallback)
                should_append = True
                if self.mode == "chat" and author and author.lower() != self.leader_name.lower():
                    should_append = False
                
                if should_append:
                    self.conversation_history.append({"role": "assistant", "content": ai_response})

            # Check for Assistant Summarization (Only driven by Leader messages/AI responses in Chat mode, or all in Personal)
            should_summarize = False
            if self.summarization_frequency > 0:
                if self.mode == "personal":
                    self.summarization_counter += 1
                    if self.summarization_counter >= self.summarization_frequency:
                        should_summarize = True
                elif self.mode == "chat":
                    # In chat mode, we only count leader interactions? 
                    # "los resúmenes del asistente se realizarán únicamente a partir de los mensajes del líder... y sus respuestas"
                    # We increment counter every time functionality is called? 
                    # Or only when leader speaks?
                    # Let's assume we increment on every AI generation, but the *content* of summary is filtered.
                    # Actually, if conversation_history only contains Leader+AI, then standard summarization works fine on it.
                    self.summarization_counter += 1
                    if self.summarization_counter >= self.summarization_frequency:
                        should_summarize = True

            if should_summarize:
                print("Performing summarization...")
                self.perform_summarization(self.auto_save)
                self.summarization_counter = 0

            # Forzar guardado en modo chat si está activado auto_save, para persistir el historial del líder
            elif self.mode == "chat" and self.auto_save:
                self.save_status(update_content=False)

            # Persist AI response for the user who triggered it (if not leader)
            if self.mode == "chat" and author and author.lower() != self.leader_name.lower():
                self._update_chat_user_memory(author, ai_response, self.personality_name, timestamp)

            return ai_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def _update_chat_user_memory(self, username, message, author, timestamp):
        """
        Maneja la memoria de un usuario específico en modo chat.
        Sirve tanto para mensajes del usuario como respuestas de la IA.
        """
        if not self.memories_manager:
            return "", []

        user_data = self.memories_manager.get_user_memory(username)
        if not user_data:
            user_data = {
                "last_messages": [],
                "summary": "",
                "counter": 10
            }

        # Update messages
        user_data["last_messages"].append({
            "content": message,
            "timestamp": str(timestamp or datetime.now()),
            "author": author # Use the passed author (User or AI)
        })
        # Keep only last 10
        if len(user_data["last_messages"]) > 10:
            user_data["last_messages"] = user_data["last_messages"][-10:]

        # Update counter
        user_data["counter"] -= 1

        # Check for summarization
        if user_data["counter"] <= 0:
            self._summarize_user(username, user_data)
            user_data["counter"] = 10
        
        # Save updates
        self.memories_manager.update_user_memory(username, user_data)
        
        return user_data["summary"], user_data["last_messages"]

    def _summarize_user(self, username, user_data):
        """Genera un resumen para un usuario específico"""
        print(f"Resumiendo usuario {username}...")
        try:
             # Construct prompt
            previous_summary = user_data.get("summary", "")
            # Include Author in the text representation
            messages_text = "\n".join([f"[{m['timestamp']}] {m.get('author', 'Unknown')}: {m['content']}" for m in user_data['last_messages']])
            
            prompt_text = f"Genera un perfil psicológico y resumen breve sobre el usuario {username} basado en sus mensajes recientes. " \
                          f"Combina esto con lo que ya sabías de él.\n\n" \
                          f"Resumen previo: {previous_summary}\n\n" \
                          f"Mensajes recientes:\n{messages_text}"

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}]
            )
            new_summary = completion.choices[0].message.content
            user_data["summary"] = new_summary
            print(f"Resumen actualizado para {username}")
        except Exception as e:
            print(f"Error resumiendo usuario {username}: {e}")

        
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
            # Intentar cargar el documento completo
            doc = self.memories_manager.get_personality_memory(self.personality_name)
            
            if doc and "history" in doc and isinstance(doc["history"], list):
                # Nuevo formato: lista de mensajes
                self.conversation_history = doc["history"]
                print(f"Personalidad cargada desde MongoDB (History Array): {self.personality_name}")
            elif doc and "content" in doc:
                # Formato legacy: string
                content = doc["content"]
                parts = content.split(self.FIELDS_SEPARATOR)
                if len(parts) >= 2:
                    role = parts[0]
                    content_text = parts[1]
                    if len(parts) > 2:
                        content_text += self.FIELDS_SEPARATOR + parts[2]
                    self.conversation_history.append({"role": role, "content": content_text})
                    print(f"Personalidad cargada desde MongoDB (Legacy Content): {self.personality_name}")
                    
        except Exception as e:
            print(f"Error al cargar desde MongoDB: {str(e)}")
        
    def save_status(self, update_content=True):
        """Guarda la personalidad en MongoDB"""
        if not self.memories_manager:
            print("Error: MemoriesManager no está disponible")
            return
            
        data = {
            "history": self.conversation_history
        }

        # Solo actualizamos el campo 'content' (legacy/resumen) si se solicita explícitamente
        # Normalmente esto solo debería ocurrir después de un resumen (perform_summarization)
        # o si la historia es consistente con ser solo un resumen.
        if update_content:
            content_legacy = ""
            for item in self.conversation_history:
                content_legacy += f"{item['role']}{self.FIELDS_SEPARATOR}{item['content']}\n"
            data["content"] = content_legacy
        
        # Guardar en MongoDB
        try:
            self.memories_manager.update_personality_fields(self.personality_name, data)
            if update_content:
                print(f"Personalidad (y resumen) guardada en MongoDB: {self.personality_name}")
            else:
                 print(f"Historial actualizado en MongoDB: {self.personality_name}")
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