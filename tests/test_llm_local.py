import requests
import json

def preguntar_ollama(mensaje, modelo="gemma3:4b"):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": modelo,
        "messages": [
            {"role": "user", "content": mensaje}
        ],
        "stream": False  # Desactivamos el streaming para obtener una única respuesta
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        respuesta = response.json()
        return respuesta.get("message", {}).get("content", "Sin respuesta")
    
    except requests.exceptions.ConnectionError:
        return "Error: No se pudo conectar al servidor de Ollama. Asegúrate de que Ollama esté ejecutándose."
    except requests.exceptions.HTTPError as e:
        return f"Error HTTP: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error decodificando JSON: {str(e)}\nRespuesta raw: {response.text}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

# Ejemplo de uso
print("Intentando conectar con Ollama...")
respuesta = preguntar_ollama("¿Cuál es la capital de Francia?")
print("Respuesta del modelo:", respuesta)
