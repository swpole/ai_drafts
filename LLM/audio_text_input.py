import gradio as gr
import speech_recognition as sr
from typing import Optional, Callable
import tempfile
import os
import whisper
import warnings

# Игнорируем предупреждения Whisper
warnings.filterwarnings("ignore")

class AudioTextInput:
    def __init__(self, on_insert_callback: Optional[Callable] = None):
        self.recognizer = sr.Recognizer()
        self.on_insert_callback = on_insert_callback
        self.current_text = ""
        self.cursor_position = 0
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Загрузка модели Whisper (ленивая загрузка)"""
        if self.whisper_model is None:
            try:
                self.whisper_model = whisper.load_model("base")
            except Exception as e:
                print(f"Ошибка загрузки модели Whisper: {e}")
                return False
        return True
    
    def transcribe_audio_speechrecognition(self, audio_path: str) -> str:
        """Преобразование аудио в текст с помощью SpeechRecognition (Google)"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='ru-RU')
                return text
        except Exception as e:
            return f"Ошибка распознавания SpeechRecognition: {str(e)}"
    
    def transcribe_audio_whisper(self, audio_path: str) -> str:
        """Преобразование аудио в текст с помощью Whisper"""
        if not self.load_whisper_model():
            return "Ошибка: модель Whisper не загружена"
        
        try:
            # Whisper может работать напрямую с аудиофайлами
            result = self.whisper_model.transcribe(audio_path, language='ru')
            return result["text"]
        except Exception as e:
            return f"Ошибка распознавания Whisper: {str(e)}"
    
    def transcribe_audio(self, audio_path: str, method: str) -> str:
        """Преобразование аудио в текст выбранным методом"""
        if not audio_path:
            return "Аудио не найдено"
            
        if method == "Google Speech Recognition":
            return self.transcribe_audio_speechrecognition(audio_path)
        elif method == "Whisper":
            return self.transcribe_audio_whisper(audio_path)
        else:
            return "Неизвестный метод распознавания"
    
    def create_interface(self):
        """Создание интерфейса Gradio"""
        with gr.Blocks(title="Аудио в текст") as self.interface:
            
            gr.Markdown("# Преобразование речи в текст")
            gr.Markdown("Выберите метод распознавания и загрузите/запишите аудио")
            
            with gr.Row():
                # Выбор метода распознавания
                method_selector = gr.Dropdown(
                    choices=["Google Speech Recognition", "Whisper"],
                    value="Google Speech Recognition",
                    label="Метод распознавания",
                    info="Выберите движок для преобразования речи в текст"
                )
                
                # Информация о методах
                with gr.Column():
                    gr.Markdown("""
                    **О методах:**
                    - **Google Speech Recognition**: Онлайн-распознавание через Google API
                    - **Whisper**: Локальное распознавание от OpenAI (требует загрузки модели)
                    """)
            
            with gr.Row():
                # Аудио компонент для записи/загрузки
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Запись аудио",
                    waveform_options={"show_controls": True}
                )
                
            with gr.Row():
                # Кнопка преобразования
                transcribe_btn = gr.Button("Преобразовать в текст", variant="primary")
                clear_btn = gr.Button("Очистить")
                
            with gr.Row():
                # Текстовое поле для отображения и редактирования
                text_transcribed = gr.Textbox(
                    label="Распознанный текст",
                    lines=5,
                    interactive=True,
                    show_copy_button=True,
                    placeholder="Текст появится здесь после преобразования...",
                    info="Выделите весь текст Ctrl+A для копирования"
                )

            with gr.Row():
                # Текстовое поле для полного текста
                text_summary = gr.Textbox(
                    label="Полный текст",
                    lines=8,
                    interactive=True,
                    show_copy_button=True,
                    placeholder="Здесь можно собрать весь распознанный текст..."
                )
            
            # Обработчики событий
            transcribe_btn.click(
                fn=self._handle_transcription,
                inputs=[audio_input, method_selector],
                outputs=[text_transcribed]
            )
            
            clear_btn.click(
                fn=lambda: ["", ""],
                inputs=[],
                outputs=[text_transcribed, text_summary]
            )
            
            # Автоматическое обновление полного текста при новом распознавании
            text_transcribed.change(
                fn=self._update_summary,
                inputs=[text_transcribed, text_summary],
                outputs=[text_summary]
            )

        return self.interface
        
    
    def _handle_transcription(self, audio_path: str, method: str) -> str:
        """Обработка преобразования аудио в текст"""
        return self.transcribe_audio(audio_path, method)
    
    def _update_summary(self, new_text: str, current_summary: str) -> str:
        """Обновление полного текста новым распознанным текстом"""
        if not new_text or new_text.startswith("Ошибка") or new_text.startswith("Аудио не найдено"):
            return current_summary
        
        if current_summary:
            return current_summary + "\n\n" + new_text
        else:
            return new_text

    def launch(self, **kwargs):
        """Запуск интерфейса"""
        self.create_interface()
        return self.interface.launch(**kwargs)

# Пример использования класса
def main():
    # Создание и запуск интерфейса
    audio_input = AudioTextInput()
    
    # Предупреждение о первом запуске Whisper
    print("Примечание: При первом использовании Whisper будет загружена модель (это может занять некоторое время)")
    print("Запуск интерфейса...")
    
    audio_input.launch(share=True)

if __name__ == "__main__":
    main()