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

    # Métodos para gestionar memorias
    def has_memory(self, memory_name: str) -> bool:
        """
        Verifica si existe un documento de memoria en MongoDB
        
        Args:
            memory_name: Nombre de la memoria
            
        Returns:
            True si existe el documento, False en caso contrario
        """
        try:
            return self.memories_collection.find_one({"_id": memory_name}) is not None
        except Exception as e:
            print(f"Error al verificar memoria en MongoDB: {str(e)}")
            return False
    
    def load_memory(self, memory_name: str):
        """
        Carga la memoria desde MongoDB
        
        Args:
            memory_name: Nombre de la memoria
            
        Returns:
            Documento de memoria
        """
        try:
            return self.memories_collection.find_one({"_id": memory_name})
        except Exception as e:
            print(f"Error al cargar memoria desde MongoDB: {str(e)}")
            return None
    
    def save_memory(self, memory_name: str, memory_summary: str, memory_latest_messages_history: list, memory_time_to_sleep: int):
        """
        Guarda o actualiza la memoria en MongoDB
        
        Args:
            memory_name: Nombre de la memoria
            memory_summary: Resumen de la memoria
            memory_latest_messages_history: Historial de mensajes
            memory_time_to_sleep: Tiempo de espera entre respuestas
        """
        try:
            self.memories_collection.update_one(
                {"_id": memory_name},
                {"$set": {"summary": memory_summary, "latest_messages_history": memory_latest_messages_history, "time_to_sleep": memory_time_to_sleep}},
                upsert=True
            )
        except Exception as e:
            print(f"Error al guardar memoria en MongoDB: {str(e)}")

    def create_memory(self, memory_name: str, time_to_sleep: int):
        """
        Crea una nueva memoria en MongoDB
        
        Args:
            memory_name: Nombre de la memoria
            time_to_sleep: Número de iteraciones necesarias antes de realizar un resumen
        """
        try:
            self.memories_collection.insert_one(
                {"_id": memory_name, "summary": "", "latest_messages_history": [], "time_to_sleep": time_to_sleep}
            )
        except Exception as e:
            print(f"Error al crear memoria en MongoDB: {str(e)}")
    
    # Métodos para gestionar personalidades
    def has_personality(self, personality_name: str) -> bool:
        """
        Verifica si existe un documento de personalidad en MongoDB
        
        Args:
            personality_name: Nombre de la personalidad
            
        Returns:
            True si existe el documento, False en caso contrario
        """
        try:
            return self.personalities_collection.find_one({"_id": personality_name}) is not None
        except Exception as e:
            print(f"Error al verificar personalidad en MongoDB: {str(e)}")
            return False
    
    def load_personality(self, personality_name: str):
        """
        Carga la personalidad desde MongoDB
        
        Args:
            personality_name: Nombre de la personalidad
            
        Returns:
            Documento de personalidad
        """
        try:
            return self.personalities_collection.find_one({"_id": personality_name})
        except Exception as e:
            print(f"Error al cargar personalidad desde MongoDB: {str(e)}")
            return None
    
    def save_personality(self, personality_name: str, personality_history: list, personality_summary: str, personality_tts: int):
        """
        Guarda o actualiza la personalidad en MongoDB
        
        Args:
            personality_name: Nombre de la personalidad
            personality_history: Historial de la personalidad
            personality_summary: Resumen de la personalidad
            personality_tts: Tiempo de espera entre respuestas de la personalidad
        """
        try:
            self.personalities_collection.update_one(
                {"_id": personality_name},
                {"$set": {"latest_messages_history": personality_history, "summary": personality_summary, "time_to_sleep": personality_tts}},
                upsert=True
            )
        except Exception as e:
            print(f"Error al guardar personalidad en MongoDB: {str(e)}")

    def create_personality(self, personality_name: str, time_to_sleep: int):
        """
        Crea una nueva personalidad en MongoDB
        
        Args:
            personality_name: Nombre de la personalidad
            time_to_sleep: Número de iteraciones necesarias antes de realizar un resumen
        """
        try:
            self.personalities_collection.insert_one(
                {"_id": personality_name, "latest_messages_history": [], "summary": "", "time_to_sleep": time_to_sleep}
            )
        except Exception as e:
            print(f"Error al crear personalidad en MongoDB: {str(e)}")
