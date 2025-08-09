# Her - Asistente de IA Modular

## Descripción del Proyecto

Este proyecto se centra en el desarrollo de un **asistente de IA modular** implementado en el módulo `ai_assistant.py`. El objetivo principal es crear una base sólida y reutilizable que permita desarrollar diferentes aplicaciones especializadas que hagan uso de esta funcionalidad central.

### Arquitectura Principal

- **`ai_assistant.py`**: Módulo principal que contiene la clase `AI_Assistant`. Esta clase proporciona:
  - Gestión de conversaciones con modelos de lenguaje (locales y en la nube)
  - Sistema de resúmenes automáticos para mantener el contexto
  - Persistencia de estado entre sesiones
  - Compatibilidad con diferentes proveedores de IA

- **Módulos Especializados**: Diferentes implementaciones que extienden la funcionalidad base:
  - **`assistants/Twitch_commentarist/bot.py`**: Bot de Twitch que combina `AI_Assistant` con capacidades de chat en vivo y síntesis de voz

## Funcionalidades Técnicas

### Texto a Voz
- Ejecutada con CUDA en `components/kokoro/`
- Síntesis de voz en tiempo real con modelo Kokoro-82M
- Soporte multiidioma configurable:
  - **Español** (voice = hf_beta)
  - **Inglés** (voice = bf_emma)
  - **Francés** (language = french)
  - **Italiano** (language = italian)
  - **Japonés** (language = japanese)
- Configuración de idioma y voz en la sección `[VOICE]` del archivo `config.ini`

### Integración con Twitch
- Bot receptor de mensajes implementado en `assistants/Twitch_commentarist/bot.py`
- Respuestas automáticas generadas por IA
- Sistema de personalidades con persistencia en archivos `.her`

### Gestión de Modelos IA
- Soporte para modelos locales (Ollama) y en la nube (OpenAI, DeepSeek, etc.)
- Configuración flexible mediante parámetros de inicialización

## Configuración Inicial

1. **Copia el archivo de configuración**: `cp config.ini.example config.ini`
2. **Edita `config.ini`** con tus credenciales reales
3. **Configura tu modelo de IA** (local con Ollama o en la nube)
4. **Configura la síntesis de voz** en la sección `[VOICE]`:
   - `voice`: Selecciona la voz específica (hf_beta para español, bf_emma para inglés)
   - `language`: Define el idioma (spanish, english, french, italian, japanese)
5. **Añade tus credenciales de Twitch** si usas el bot comentarista

## Modos de Funcionamiento

### Modo Twitch Commentarist
- El bot recibe mensajes de Twitch y responde con respuestas generadas por la IA
- Incluye síntesis de voz y visualización en tiempo real
- Interfaz visual con imágenes dinámicas durante las respuestas