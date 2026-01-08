from pymongo import MongoClient


class MemoriesManager:
    def __init__(self):
        """Inicializa la conexión con MongoDB en localhost:27017"""
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['twitch']
        self.memories_collection = self.db['memories']
        self.personalities_collection = self.db['personalities']
    
    def close(self):
        """Cierra la conexión con MongoDB"""
        self.client.close()
    
    # Métodos para gestionar personalidades
    def has_personality(self, assistant_name: str) -> bool:
        """
        Verifica si existe un documento de personalidad en MongoDB
        
        Args:
            assistant_name: Nombre del asistente
            
        Returns:
            True si existe el documento, False en caso contrario
        """
        try:
            result = self.personalities_collection.find_one({"_id": assistant_name})
            return result is not None
        except Exception as e:
            print(f"Error al verificar personalidad en MongoDB: {str(e)}")
            return False
    
    def load_personality(self, assistant_name: str) -> str:
        """
        Carga la personalidad desde MongoDB
        
        Args:
            assistant_name: Nombre del asistente
            
        Returns:
            Contenido de la personalidad o None si no existe
        """
        try:
            result = self.personalities_collection.find_one({"_id": assistant_name})
            if result and "content" in result:
                return result["content"]
            return None
        except Exception as e:
            print(f"Error al cargar personalidad desde MongoDB: {str(e)}")
            return None
    
        
    def get_personality_memory(self, assistant_name: str) -> dict:
        """
        Recupera el documento completo de la personalidad.
        Args:
            assistant_name: Nombre del asistente
        Returns:
            Documento completo o None
        """
        try:
             return self.personalities_collection.find_one({"_id": assistant_name})
        except Exception as e:
            print(f"Error al recuperar personalidad completa {assistant_name}: {str(e)}")
            return None

    def save_personality(self, assistant_name: str, content: str):
        """
        Guarda o actualiza la personalidad en MongoDB
        
        Args:
            assistant_name: Nombre del asistente
            content: Contenido de la personalidad a guardar
        """
        try:
            self.personalities_collection.update_one(
                {"_id": assistant_name},
                {"$set": {"content": content}},
                upsert=True
            )
        except Exception as e:
            print(f"Error al guardar personalidad en MongoDB: {str(e)}")

    def update_personality_fields(self, assistant_name: str, data: dict):
        """
        Actualiza campos específicos de la personalidad.
        
        Args:
            assistant_name: Nombre del asistente
            data: Diccionario con campos a actualizar (ej: {"history": [...], "content": "..."})
        """
        try:
            self.personalities_collection.update_one(
                {"_id": assistant_name},
                {"$set": data},
                upsert=True
            )
        except Exception as e:
            print(f"Error al actualizar campos de personalidad {assistant_name}: {str(e)}")

    # Métodos para gestionar memorias de usuarios (Chat Mode)
    def get_user_memory(self, username: str) -> dict:
        """
        Recupera la memoria de un usuario específico.
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Diccionario con la memoria del usuario o None si no existe
        """
        try:
            return self.memories_collection.find_one({"_id": username})
        except Exception as e:
            print(f"Error al recuperar memoria de usuario {username}: {str(e)}")
            return None

    def update_user_memory(self, username: str, data: dict):
        """
        Actualiza o crea la memoria de un usuario.
        
        Args:
            username: Nombre del usuario
            data: Datos a guardar (messages, summary, counter)
        """
        try:
            self.memories_collection.update_one(
                {"_id": username},
                {"$set": data},
                upsert=True
            )
        except Exception as e:
            print(f"Error al actualizar memoria de usuario {username}: {str(e)}")
