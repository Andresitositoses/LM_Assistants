from openai import OpenAI
import os
import platform

class AI_Assistant():

    FIELDS_SEPARATOR = "|/="

    def __init__(self, initial_prompt: str, personalities_path: str, personality_name: str, summarization_frequency: int, auto_save: bool, lm_params: tuple):
        '''
        initial_prompt: str -> Prompt inicial para el asistente
        personalities_path: str -> Ruta a la carpeta de personalidades
        personality_name: str -> Nombre de la personalidad del asistente
        summarization_frequency: int -> Frecuencia de resumen (en conversaciones). < 0 -> No se realiza ningún resumen.
        auto_save: bool -> Si se debe guardar el estado del asistente en un archivo de texto.
        lm_params: tuple -> (base_url, api_key, model, is_local) - Parámetros del modelo de lenguaje
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
        
        # Configurar el directorio de personalidades
        self.personalities_path = personalities_path

        # Conversation history
        self.conversation_history = []
        if self.has_status():
            self.load_status()
        else:
            self.conversation_history.append({"role": "system", "content": self.initial_prompt})
        
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
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history
            )
            ai_response = completion.choices[0].message.content

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
            text = "Hazte un resumen mínimo de los aspectos más relevantes de la conversación que has mantenido actualmente y lo aprendido en conversaciones anteriores, con el fin de poder recordarlos más adelante."
            text += f"\n\nEn conversaciones anteriores: {self.conversation_history[0]['content'].split(self.FIELDS_SEPARATOR)[1]}" if len(self.conversation_history[0]['content'].split(self.FIELDS_SEPARATOR)) >= 2 else ""
            self.conversation_history.append({"role": "user", "content": text})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history
            )
            ai_response = completion.choices[0].message.content

            self.conversation_history.clear()
            self.conversation_history.append({"role": "system", "content": f"{self.initial_prompt}. Aprendido en conversaciones anteriores:{self.FIELDS_SEPARATOR} {ai_response}"})
            print(f"\n\n(System) {self.conversation_history[-1]['content']}\n\n")

            if save:
                self.save_status()

        except Exception as e:
            print(f"Error: {str(e)}")
            return None
        
    def load_status(self):
        try:
            file_path = os.path.join(self.personalities_path, f"{self.personality_name}.her")
            with open(file_path, "r") as File:
                content = File.read()
                # Dividir por el separador principal
                parts = content.split(self.FIELDS_SEPARATOR)
                if len(parts) >= 2:
                    # El primer elemento es el rol (system)
                    role = parts[0]
                    # El segundo elemento es el contenido
                    content = parts[1]
                    # Si hay más partes, son parte del contenido
                    if len(parts) > 2:
                        content += self.FIELDS_SEPARATOR + parts[2]
                    self.conversation_history.append({"role": role, "content": content})
        except Exception as e:
            print(f"Error al cargar el estado: {str(e)}")
        
    def save_status(self):
        try:
            if not os.path.exists(self.personalities_path):
                os.makedirs(self.personalities_path)
                
            file_path = os.path.join(self.personalities_path, f"{self.personality_name}.her")
            with open(file_path, "w") as File:
                for item in self.conversation_history:
                    File.write(f"{item['role']}{self.FIELDS_SEPARATOR}{item['content']}\n")
                    
        except Exception as e:
            print(f"Error al guardar el estado: {str(e)}")

    def has_status(self) -> bool:
        try:
            file_path = os.path.join(self.personalities_path, f"{self.personality_name}.her")
            with open(file_path, "r") as File:
                pass
            return True
        except Exception as e:
            return False