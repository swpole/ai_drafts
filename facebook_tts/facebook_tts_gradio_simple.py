# simple_gradio_tts.py (альтернативная упрощенная версия)
import gradio as gr
from facebook_tts_class import TextToSpeech  # Импортируем наш класс
import tempfile

# Доступные модели
MODELS = {
    "Русский": "facebook/mms-tts-rus",
    "Английский": "facebook/mms-tts-eng", 
    "Французский": "facebook/mms-tts-fra"
}

tts = TextToSpeech()

def generate_speech(text, language):
    if not text.strip():
        return None, "Введите текст"
    
    model_name = MODELS[language]
    tts.load_model(model_name)
    
    audio_path, message = tts.generate_speech(text)
    return audio_path

# Простой интерфейс
interface = gr.Interface(
    fn=generate_speech,
    inputs=[
        gr.Textbox(label="Текст", lines=3),
        gr.Dropdown(choices=list(MODELS.keys()), label="Язык", value="Русский")
    ],
    outputs=gr.Audio(label="Результат"),
    title="Text to Speech",
    description="Преобразование текста в речь с выбором языка"
)

if __name__ == "__main__":
    interface.launch()