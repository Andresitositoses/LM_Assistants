import requests
import json
import base64
import os
import argparse
from pathlib import Path

def cargar_imagen_base64(ruta_imagen):
    """
    Carga una imagen y la convierte a base64
    """
    try:
        with open(ruta_imagen, "rb") as imagen_file:
            imagen_data = imagen_file.read()
            imagen_base64 = base64.b64encode(imagen_data).decode('utf-8')
            return imagen_base64
    except Exception as e:
        print(f"Error cargando imagen {ruta_imagen}: {e}")
        return None

def analizar_imagen_ollama(ruta_imagen, prompt="Describe esta imagen", modelo="gemma3:4b"):
    """
    Analiza una imagen usando Ollama
    """
    # Cargar imagen en base64
    imagen_base64 = cargar_imagen_base64(ruta_imagen)
    if not imagen_base64:
        return "Error: No se pudo cargar la imagen"

    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": modelo,
        "prompt": prompt,
        "images": [imagen_base64],
        "stream": False
    }

    try:
        print(f"Analizando imagen: {ruta_imagen}")
        print(f"Prompt: {prompt}")
        print("Enviando solicitud...")
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        respuesta = response.json()
        return respuesta.get("response", "Sin respuesta")
        
    except requests.exceptions.ConnectionError:
        return "Error: No se pudo conectar al servidor de Ollama. Asegúrate de que Ollama esté ejecutándose."
    except requests.exceptions.HTTPError as e:
        return f"Error HTTP: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error decodificando JSON: {str(e)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

def analizar_imagen_especifica(ruta_imagen, prompt=None, modelo="gemma3:4b"):
    """
    Analiza una imagen específica pasada como parámetro
    """
    # Verificar que la imagen existe
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: La imagen '{ruta_imagen}' no existe")
        return
    
    # Verificar que es un archivo de imagen
    extensiones_validas = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    extension = Path(ruta_imagen).suffix.lower()
    
    if extension not in extensiones_validas:
        print(f"❌ Error: '{extension}' no es una extensión de imagen válida")
        print(f"Extensiones soportadas: {', '.join(extensiones_validas)}")
        return
    
    print(f"🖼️  Imagen: {ruta_imagen}")
    print(f"📐 Tamaño: {os.path.getsize(ruta_imagen)} bytes")
    print(f"🤖 Modelo: {modelo}")
    print("-" * 50)
    
    # Si no se especifica prompt, usar uno por defecto
    if not prompt:
        prompt = "Describe esta imagen en detalle"
    
    # Analizar imagen
    resultado = analizar_imagen_ollama(ruta_imagen, prompt, modelo)
    
    print(f"\n📋 Análisis:")
    print(resultado)

def listar_imagenes_disponibles():
    """
    Lista todas las imágenes disponibles en el directorio img/
    """
    extensiones_imagen = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    imagenes = []
    
    img_dir = Path("img")
    if img_dir.exists():
        for archivo in img_dir.rglob("*"):
            if archivo.is_file() and archivo.suffix.lower() in extensiones_imagen:
                imagenes.append(str(archivo))
    
    return imagenes

def test_analisis_imagenes():
    """
    Prueba el análisis de varias imágenes con diferentes prompts
    """
    print("=== TEST DE ANÁLISIS DE IMÁGENES CON OLLAMA ===")
    print(f"Modelo: gemma3:4b")
    print("-" * 50)
    
    # Obtener lista de imágenes
    imagenes = listar_imagenes_disponibles()
    
    if not imagenes:
        print("No se encontraron imágenes en el directorio img/")
        return
    
    print(f"Imágenes encontradas: {len(imagenes)}")
    for i, img in enumerate(imagenes[:5], 1):  # Mostrar solo las primeras 5
        print(f"{i}. {img}")
    
    # Prompts de prueba
    prompts_prueba = [
        "Describe esta imagen en detalle",
        "¿Qué ves en esta imagen?",
        "Analiza los colores y la composición de esta imagen",
        "¿Qué emociones transmite esta imagen?"
    ]
    
    # Probar con la primera imagen disponible
    imagen_prueba = imagenes[0]
    print(f"\n--- Analizando: {imagen_prueba} ---")
    
    for i, prompt in enumerate(prompts_prueba, 1):
        print(f"\n🔍 Análisis {i}:")
        resultado = analizar_imagen_ollama(imagen_prueba, prompt)
        print(f"Resultado: {resultado}")
        print("-" * 30)

def analisis_interactivo():
    """
    Permite al usuario seleccionar una imagen y hacer preguntas sobre ella
    """
    print("\n=== ANÁLISIS INTERACTIVO ===")
    
    # Listar imágenes disponibles
    imagenes = listar_imagenes_disponibles()
    
    if not imagenes:
        print("No se encontraron imágenes en el directorio img/")
        return
    
    print("Imágenes disponibles:")
    for i, img in enumerate(imagenes, 1):
        print(f"{i}. {img}")
    
    try:
        seleccion = int(input(f"\nSelecciona una imagen (1-{len(imagenes)}): ")) - 1
        
        if 0 <= seleccion < len(imagenes):
            imagen_seleccionada = imagenes[seleccion]
            print(f"Imagen seleccionada: {imagen_seleccionada}")
            
            while True:
                prompt_usuario = input("\n¿Qué quieres saber sobre esta imagen? (o 'salir' para terminar): ")
                
                if prompt_usuario.lower() == 'salir':
                    break
                
                resultado = analizar_imagen_ollama(imagen_seleccionada, prompt_usuario)
                print(f"\n📋 Análisis: {resultado}")
        else:
            print("Selección inválida")
            
    except ValueError:
        print("Por favor, introduce un número válido")
    except KeyboardInterrupt:
        print("\nSaliendo...")

def check_vision_models():
    """
    Verifica qué modelos con capacidades de visión están disponibles
    """
    try:
        url = "http://localhost:11434/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        
        models = response.json()
        print("Modelos disponibles:")
        vision_keywords = ['llava', 'vision', 'multimodal', 'gemma']
        
        for model in models.get("models", []):
            model_name = model.get('name', '').lower()
            is_vision = any(keyword in model_name for keyword in vision_keywords)
            status = "🔍 (Posible capacidad de visión)" if is_vision else ""
            print(f"- {model.get('name', 'Sin nombre')} {status}")
            
    except Exception as e:
        print(f"Error verificando modelos: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Analiza imágenes usando Ollama con el modelo gemma3:4b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python test_image_analysis.py --imagen "img/Perfectas/p1.png"
  python test_image_analysis.py --imagen "img/Perfectas/p1.png" --prompt "¿Qué colores predominan?"
  python test_image_analysis.py --imagen "img/Perfectas/p1.png" --modelo "gemma3:27b"
  python test_image_analysis.py --test                    # Ejecutar tests automáticos
  python test_image_analysis.py --interactivo            # Modo interactivo
        """
    )
    
    parser.add_argument(
        '--imagen', '-i',
        type=str,
        help='Ruta de la imagen a analizar'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default="Describe esta imagen en detalle",
        help='Prompt para el análisis (por defecto: "Describe esta imagen en detalle")'
    )
    
    parser.add_argument(
        '--modelo', '-m',
        type=str,
        default="gemma3:4b",
        help='Modelo de Ollama a usar (por defecto: gemma3:4b)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Ejecutar tests automáticos con imágenes del directorio img/'
    )
    
    parser.add_argument(
        '--interactivo', '-int',
        action='store_true',
        help='Modo interactivo para seleccionar imágenes'
    )
    
    parser.add_argument(
        '--listar', '-l',
        action='store_true',
        help='Listar todas las imágenes disponibles en el directorio img/'
    )
    
    args = parser.parse_args()
    
    # Si no se especifica ninguna acción, mostrar ayuda
    if not any([args.imagen, args.test, args.interactivo, args.listar]):
        parser.print_help()
        return
    
    # Verificar modelos disponibles
    if args.test or args.interactivo:
        print("Verificando modelos disponibles...")
        check_vision_models()
        print()
    
    # Ejecutar según los parámetros
    if args.listar:
        imagenes = listar_imagenes_disponibles()
        print(f"📁 Imágenes encontradas ({len(imagenes)}):")
        for img in imagenes:
            print(f"  - {img}")
    
    elif args.imagen:
        analizar_imagen_especifica(args.imagen, args.prompt, args.modelo)
    
    elif args.test:
        test_analisis_imagenes()
    
    elif args.interactivo:
        analisis_interactivo()

if __name__ == "__main__":
    main() 