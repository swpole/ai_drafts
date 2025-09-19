import torch
from scipy.io import wavfile
from transformers import VitsModel, AutoTokenizer
from pathlib import Path
from typing import Optional, Union
import logging
import gradio as gr
import os
import tempfile

class TextToSpeech:
    def __init__(self, 
                 model_name: str = "facebook/mms-tts-rus",
                 device: Optional[str] = None):
        """
        Инициализация модели преобразования текста в речь
        """
        self.model_name = model_name
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def _setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self, model_name: str):
        """Загрузка модели и токенизатора"""
        try:
            self.logger.info(f"Загрузка модели {model_name} на устройство {self.device}")
            self.model = VitsModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_name = model_name
            self.is_loaded = True
            self.logger.info("Модель успешно загружена")
            return f"Модель {model_name} загружена успешно!"
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            return f"Ошибка загрузки модели: {e}"
    
    def generate_speech(self, 
                       text: str,
                       sampling_rate: Optional[int] = None,
                       normalize_audio: bool = True) -> tuple:
        """
        Генерация речи из текста и возврат временного файла
        """
        if not self.is_loaded:
            return None, "Сначала загрузите модель!"
        
        try:
            self.logger.info(f"Генерация речи для текста: {text[:50]}...")
            
            # Токенизация текста
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Генерация аудио
            with torch.no_grad():
                output = self.model(**inputs, speaker_id=1).waveform
            
            # Создание временного файла
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Сохранение аудио
            audio = output.cpu().numpy()
            audio = audio.squeeze()
            
            if normalize_audio:
                audio = (audio * 32767).astype('int16')
            
            sr = sampling_rate if sampling_rate else self.model.config.sampling_rate
            wavfile.write(temp_path, sr, audio)
            
            return temp_path, "Аудио успешно сгенерировано!"
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации речи: {e}")
            return None, f"Ошибка генерации: {e}"

# Список доступных моделей
AVAILABLE_MODELS = {
    "Russian (Facebook MMS)": "facebook/mms-tts-rus",
    "Russian (multispeaker)": "utrobinmv/tts_ru_free_hf_vits_high_multispeaker",
    "English (Facebook MMS)": "facebook/mms-tts-eng",
    "French (Facebook MMS)": "facebook/mms-tts-fra",
    "German (Facebook MMS)": "facebook/mms-tts-deu",
    "Spanish (Facebook MMS)": "facebook/mms-tts-spa",
    "Italian (Facebook MMS)": "facebook/mms-tts-ita",
}

# Глобальный экземпляр TTS
tts_engine = TextToSpeech()

def load_selected_model(model_name):
    """Загрузка выбранной модели"""
    model_id = AVAILABLE_MODELS[model_name]
    message = tts_engine.load_model(model_id)
    return message

def generate_audio(text, model_name):
    """Генерация аудио из текста"""
    if not text.strip():
        return None, "Введите текст для преобразования"
    
    # Если модель не загружена или выбрана другая, загружаем её
    if not tts_engine.is_loaded or AVAILABLE_MODELS[model_name] != tts_engine.model_name:
        load_message = load_selected_model(model_name)
        if "Ошибка" in load_message:
            return None, load_message
    
    audio_path, message = tts_engine.generate_speech(text)
    return audio_path, message

def cleanup_temp_files():
    """Очистка временных файлов при завершении"""
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith('.wav') and file.startswith('tmp'):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass

# Создание Gradio интерфейса
with gr.Blocks(title="Text to Speech Converter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Text to Speech Converter")
    gr.Markdown("Преобразуйте текст в естественную речь с помощью AI моделей")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value="Russian (Facebook MMS)",
                label="Выберите модель",
                info="Выберите язык и модель для синтеза речи"
            )
            
            load_btn = gr.Button("🔄 Загрузить модель", variant="primary")
            load_status = gr.Textbox(label="Статус модели", interactive=False)
        
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Введите текст",
                placeholder="Введите текст для преобразования в речь...",
                lines=4,
                max_lines=10
            )
            
            generate_btn = gr.Button("🎵 Сгенерировать аудио", variant="primary")
            
            audio_output = gr.Audio(
                label="Сгенерированное аудио",
                type="filepath",
                interactive=False
            )
            
            status_output = gr.Textbox(label="Статус генерации", interactive=False)
    
    # Обработчики событий
    load_btn.click(
        fn=load_selected_model,
        inputs=model_dropdown,
        outputs=load_status
    )
    
    generate_btn.click(
        fn=generate_audio,
        inputs=[text_input, model_dropdown],
        outputs=[audio_output, status_output]
    )
    
    # Автозагрузка модели при изменении выбора
    model_dropdown.change(
        fn=load_selected_model,
        inputs=model_dropdown,
        outputs=load_status
    )
    
    # Примеры текста
    gr.Examples(
        examples=[
            ["Привет! Как твои дела? Это тест преобразования текста в речь."],
            ["Ночью двадцать тр+етьего июня начал извергаться самый высокий \
действующий вулк+ан в Евразии - Кл+ючевской. Об этом сообщила руководитель \
Камчатской группы реагирования на вулканические извержения, ведущий \
научный сотрудник Института вулканологии и сейсмологии ДВО РАН +Ольга Гирина.\
«Зафиксированное ночью не просто свечение, а вершинное эксплозивное \
извержение стромболианского типа. Пока такое извержение никому не опасно: \
ни населению, ни авиации» пояснила ТАСС госпожа Гирина."],
            ["Hello! How are you? This is a text-to-speech conversion test."],
            ["Bonjour! Comment ça va? Ceci est un test de synthèse vocale."],
            ["Hola! ¿Cómo estás? Esta es una prueba de texto a voz."]
        ],
        inputs=text_input,
        label="Примеры текста для тестирования"
    )
    
    # Информация о моделях
    gr.Markdown("""
    ## 📋 Информация о моделях
    
    - **Russian (Facebook MMS)** - Модель для русского языка
    - **English (Facebook MMS)** - Модель для английского языка  
    - **French (Facebook MMS)** - Модель для французского языка
    - **German (Facebook MMS)** - Модель для немецкого языка
    - **Spanish (Facebook MMS)** - Модель для испанского языка
    - **Italian (Facebook MMS)** - Модель для итальянского языка
    
    *Модели автоматически загружаются при выборе*
    """)

# Запуск приложения
if __name__ == "__main__":
    # Автоматическая загрузка первой модели при запуске
    initial_model = list(AVAILABLE_MODELS.keys())[0]
    tts_engine.load_model(AVAILABLE_MODELS[initial_model])
    
    # Запуск интерфейса
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
    
    # Очистка при завершении
    import atexit
    atexit.register(cleanup_temp_files)