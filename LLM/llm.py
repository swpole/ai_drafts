import gradio as gr
import whisper
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import subprocess
import os
import tempfile
import json
import requests

# Инициализация модели для транскрибации
try:
    whisper_model = whisper.load_model("base")
    print("Модель Whisper загружена успешно")
except Exception as e:
    print(f"Ошибка загрузки Whisper: {e}")
    whisper_model = None

def get_installed_ollama_models():
    """
    Автоматически получает список установленных моделей Ollama через API.
    Возвращает список названий моделей.
    """
    models_list = []
    
    try:
        # Попробуем получить модели через API Ollama
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models_list = [model["name"] for model in data.get("models", [])]
            print(f"Найдены модели через API: {models_list}")
            return models_list
    except requests.exceptions.RequestException as e:
        print(f"API Ollama недоступно: {e}")
        # Пробуем альтернативный способ через командную строку
        pass
    
    try:
        # Альтернативный способ: через команду ollama list
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Пропускаем заголовок
                if line.strip():
                    model_name = line.split()[0]
                    models_list.append(model_name)
            print(f"Найдены модели через CLI: {models_list}")
            return models_list
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Команда ollama list не сработала: {e}")
        pass
    
    # Если ничего не работает, возвращаем пустой список или дефолтные модели
    print("Не удалось автоматически определить модели. Использую дефолтный список.")
    return ["llama3.1", "gemma2", "mistral"]

def check_ollama_running():
    """
    Проверяет, запущен ли сервер Ollama.
    """
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama_server():
    """
    Пытается запустить сервер Ollama.
    """
    try:
        print("Попытка запуска сервера Ollama...")
        # Запускаем в фоновом режиме (зависит от ОС)
        if os.name == 'nt':  # Windows
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:  # Linux/Mac
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        # Даем серверу время на запуск
        import time
        time.sleep(3)
        return check_ollama_running()
    except Exception as e:
        print(f"Ошибка запуска сервера Ollama: {e}")
        return False

def record_audio(duration=5, sample_rate=16000):
    """
    Записывает аудио с микрофона указанной длительности.
    Возвращает путь к временному файлу в формате WAV.
    """
    print(f"Запись аудио в течение {duration} секунд...")
    try:
        # Запись аудио
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='int16')
        sd.wait()  # Ждем окончания записи

        # Создаем временный файл для сохранения
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, sample_rate)
        print(f"Аудио сохранено в: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        print(f"Ошибка записи аудио: {e}")
        return None

def transcribe_audio(audio_file_path):
    """
    Преобразует аудиофайл в текст с помощью модели Whisper.
    Возвращает распознанный текст.
    """
    if audio_file_path is None:
        return "Аудиофайл не предоставлен."
    
    if whisper_model is None:
        return "Модель Whisper не загружена. Проверьте установку."

    print(f"Транскрибация аудио: {audio_file_path}")
    try:
        # Загружаем аудио и применяем модель
        result = whisper_model.transcribe(audio_file_path, language="ru")
        transcribed_text = result["text"]
        print(f"Распознанный текст: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        return f"Ошибка транскрибации: {str(e)}"

def query_ollama(model_name, prompt):
    """
    Отправляет запрос (prompt) в выбранную модель Ollama и возвращает ответ.
    """
    if not prompt.strip():
        return "Запрос пуст. Введите текст или преобразуйте аудио."
    
    # Проверяем, запущен ли сервер Ollama
    if not check_ollama_running():
        if not start_ollama_server():
            return "Сервер Ollama не запущен. Запустите Ollama вручную."

    print(f"Отправка запроса модели {model_name}: {prompt}")
    try:
        # Используем API Ollama для более надежного взаимодействия
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, 
                               timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Ответ не получен")
        else:
            return f"Ошибка API: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Таймаут запроса к модели. Попробуйте позже."
    except requests.exceptions.RequestException as e:
        return f"Ошибка соединения с Ollama: {str(e)}"
    except Exception as e:
        return f"Неожиданная ошибка: {str(e)}"

def refresh_models():
    """
    Обновляет список доступных моделей.
    """
    return gr.Dropdown(choices=get_installed_ollama_models())

def create_gradio_interface():
    """
    Создает и запускает веб-интерфейс Gradio.
    """
    # Получаем список моделей при запуске
    available_models = get_installed_ollama_models()
    
    with gr.Blocks(title="Голосовой LLM ассистент", theme="soft") as demo:
        gr.Markdown("# 🎙️ Голосовой интерфейс для локальных LLM через Ollama")
        
        # Проверка статуса Ollama
        ollama_status = "🟢 Запущен" if check_ollama_running() else "🔴 Не запущен"
        gr.Markdown(f"**Статус Ollama:** {ollama_status}")

        with gr.Row():
            # Выбор модели Ollama с кнопкой обновления
            with gr.Column(scale=3):
                model_dropdown = gr.Dropdown(
                    choices=available_models, 
                    value=available_models[0] if available_models else None,
                    label="Выберите модель Ollama",
                    interactive=True
                )
            with gr.Column(scale=1):
                refresh_button = gr.Button("🔄 Обновить список моделей")

        with gr.Row():
            # Блок для работы с аудио
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Записанное или загруженное аудио", 
                    type="filepath", 
                    interactive=True
                )
                record_button = gr.Button("🎙️ Запись с микрофона (5 сек)")
                duration_slider = gr.Slider(1, 30, value=5, 
                                          label="Длительность записи (сек)")

            # Блок для работы с текстом
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Текст запроса (можно редактировать)", 
                    lines=5, 
                    placeholder="Транскрибированный текст появится здесь..."
                )
                transcribe_button = gr.Button("🔊 Преобразовать аудио в текст")
                submit_button = gr.Button("🚀 Отправить запрос LLM")

        # Блок для вывода ответа модели
        response_output = gr.Textbox(
            label="Ответ LLM", 
            lines=10, 
            interactive=False
        )

        # Обработчики событий
        # Запись аудио с настраиваемой длительностью
        record_button.click(
            fn=record_audio, 
            inputs=duration_slider,
            outputs=audio_output
        )

        # Транскрибация аудио
        transcribe_button.click(
            fn=transcribe_audio, 
            inputs=audio_output, 
            outputs=text_input
        )

        # Отправка запроса к LLM
        submit_button.click(
            fn=query_ollama, 
            inputs=[model_dropdown, text_input], 
            outputs=response_output
        )

        # Обновление списка моделей
        refresh_button.click(
            fn=refresh_models,
            outputs=model_dropdown
        )

    return demo

if __name__ == "__main__":
    # Проверяем доступность Ollama при запуске
    if not check_ollama_running():
        print("Сервер Ollama не запущен. Пытаюсь запустить...")
        if start_ollama_server():
            print("Сервер Ollama успешно запущен")
        else:
            print("Не удалось запустить сервер Ollama. Убедитесь, что Ollama установлен.")
    
    # Создаем интерфейс
    demo = create_gradio_interface()
    
    # Запускаем интерфейс
    print("Запуск приложения...")
    print("Откройте http://localhost:7860 в вашем браузере")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            share=False,
            debug=True
        )
    except Exception as e:
        print(f"Ошибка запуска приложения: {e}")
        print("Проверьте, что порт 7860 свободен или укажите другой порт")