import gradio as gr
import speech_recognition as sr
#python -m pip install speechrecognition
from typing import Optional, Callable
import tempfile
import os

class AudioTextInput:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.current_text = ""
        
    def transcribe_audio(self, audio_path: str) -> str:
        """Преобразование аудио в текст с помощью SpeechRecognition"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='ru-RU')
                return text
        except Exception as e:
            return f"Ошибка распознавания: {str(e)}"
    
    def create_interface(self):
        """Создание интерфейса Gradio"""
        with gr.Blocks(title="Аудио в текст") as self.interface:
            
            with gr.Row():
                # Аудио компонент для записи/загрузки
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Запись аудио"
                )
                
            with gr.Row():
                # Кнопка преобразования
                transcribe_btn = gr.Button("Преобразовать в текст")
                
            with gr.Row():
                # Текстовое поле для отображения и редактирования
                text_transcribed = gr.Textbox(
                    label="Распознанный текст (выделить весь текст Ctrl+A)",
                    lines=5,
                    interactive=True,
                    placeholder="Текст появится здесь после преобразования..."
                )

            with gr.Row():
                # Текстовое поле для отображения и редактирования
                text_summary = gr.Textbox(
                    label="Полный текст",
                    lines=5,
                    interactive=True,
                    placeholder="Вставляйте текст сюда..."
                )
            
            # Обработчики событий
            transcribe_btn.click(
                fn=self._handle_transcription,
                inputs=[audio_input],
                outputs=[text_transcribed]
            )
            
    
    def _handle_transcription(self, audio_path: str) -> str:
        """Обработка преобразования аудио в текст"""
        if audio_path:
            return self.transcribe_audio(audio_path)
        return "Аудио не найдено"



    def launch(self, **kwargs):
        """Запуск интерфейса"""
        self.create_interface()
        return self.interface.launch(**kwargs)

# Пример использования класса
def main():
    # Создание и запуск интерфейса
    audio_input = AudioTextInput()
    audio_input.launch(share=True)

if __name__ == "__main__":
    main()