import gradio as gr
import speech_recognition as sr
import numpy as np
from typing import Union, Tuple, Optional

class SpeechToTextApp:
    """
    Класс для преобразования речи в текст с использованием Gradio.
    Поддерживает запись с микрофона и загрузку аудиофайлов.
    """
    
    def __init__(self):
        """Инициализация распознавателя речи."""
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio: Optional[Union[str, Tuple[int, np.ndarray]]]) -> str:
        """
        Преобразует аудио в текст.
        
        Args:
            audio: Путь к файлу или кортеж (частота дискретизации, аудиоданные)
            
        Returns:
            str: Распознанный текст или сообщение об ошибке
        """
        if audio is None:
            return "Аудио не предоставлено"
        
        try:
            # Обработка в зависимости от типа входных данных
            if isinstance(audio, str):
                # Это файл
                with sr.AudioFile(audio) as source:
                    audio_data = self.recognizer.record(source)
            else:
                # Это данные с микрофона (sample_rate, numpy_array)
                sr_value, y = audio
                
                # Конвертируем в моно, если стерео
                if y.ndim > 1:
                    y = y.mean(axis=1)
                
                # Сохраняем временный файл для обработки
                import tempfile
                import wave
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    with wave.open(temp_file.name, 'wb') as wf:
                        wf.setnchannels(1)  # моно
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sr_value)
                        wf.writeframes((y * 32767).astype(np.int16).tobytes())
                    
                    with sr.AudioFile(temp_file.name) as source:
                        audio_data = self.recognizer.record(source)
            
            # Распознавание речи
            text = self.recognizer.recognize_google(audio_data, language='ru-RU')
            return text
            
        except sr.UnknownValueError:
            return "Речь не распознана"
        except sr.RequestError as e:
            return f"Ошибка сервиса распознавания: {e}"
        except Exception as e:
            return f"Произошла ошибка: {e}"
    
    def create_interface(self):
        """
        Создает Gradio интерфейс с независимым редактируемым текстовым полем.
        
        Returns:
            gr.Blocks: Графический интерфейс
        """
        with gr.Blocks(title="Преобразование речи в текст") as demo:
            gr.Markdown("# 🎤 Преобразование речи в текст")
            gr.Markdown("Запишите аудио с микрофона или загрузите аудиофайл, затем нажмите кнопку для преобразования.")
            
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Аудиовход",
                    interactive=True
                )
            
            with gr.Row():
                convert_btn = gr.Button("Преобразовать в текст", variant="primary")
            
            with gr.Row():
                # Первое текстовое поле - для распознанного текста (редактируемое)
                recognized_text = gr.Textbox(
                    label="Распознанный текст",
                    placeholder="Здесь появится распознанный текст...",
                    lines=4,
                    max_lines=10,
                    interactive=True  # Разрешаем редактирование
                )
            
            with gr.Row():
                # Независимое текстовое поле для редактирования
                editing_text = gr.Textbox(
                    label="Редактируемый текст",
                    placeholder="Здесь можно редактировать текст независимо от распознавания...",
                    lines=6,
                    max_lines=15,
                    interactive=True  # Разрешаем редактирование
                )
            
            # Обработка нажатия кнопки преобразования
            convert_btn.click(
                fn=self.transcribe_audio,
                inputs=audio_input,
                outputs=recognized_text
            )
        
        return demo
    
    def launch(self, **kwargs):
        """
        Запускает приложение.
        
        Args:
            **kwargs: Дополнительные параметры для launch()
        """
        demo = self.create_interface()
        demo.launch(**kwargs)

# Запуск приложения
if __name__ == "__main__":
    app = SpeechToTextApp()
    app.launch(share=True)