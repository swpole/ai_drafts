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

    def transcribe_audio(
        self,
        audio: Optional[Union[str, Tuple[int, np.ndarray]]],
        engine: str,
        google_cloud_key: str,
        houndify_client_id: str,
        houndify_client_key: str,
        ibm_username: str,
        ibm_password: str,
        ibm_url: str
    ) -> str:
        """
        Преобразует аудио в текст с выбранным движком.
        """
        if audio is None:
            return "Аудио не предоставлено"

        try:
            # Обработка аудиофайла или записи
            if isinstance(audio, str):
                with sr.AudioFile(audio) as source:
                    audio_data = self.recognizer.record(source)
            else:
                sr_value, y = audio
                if y.ndim > 1:
                    y = y.mean(axis=1)
                import tempfile, wave
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    with wave.open(temp_file.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr_value)
                        wf.writeframes((y * 32767).astype(np.int16).tobytes())
                    with sr.AudioFile(temp_file.name) as source:
                        audio_data = self.recognizer.record(source)

            # Выбор движка
            if engine == "Google":
                text = self.recognizer.recognize_google(audio_data, language="ru-RU")

            elif engine == "Google Cloud":
                if not google_cloud_key:
                    return "Укажите Google Cloud JSON ключ"
                text = self.recognizer.recognize_google_cloud(audio_data, credentials_json=google_cloud_key, language="ru-RU")

            elif engine == "Sphinx":
                text = self.recognizer.recognize_sphinx(audio_data, language="ru-RU")

            elif engine == "Houndify":
                if not houndify_client_id or not houndify_client_key:
                    return "Укажите Houndify Client ID и Client Key"
                text = self.recognizer.recognize_houndify(audio_data, client_id=houndify_client_id, client_key=houndify_client_key)

            elif engine == "IBM Watson":
                if not ibm_username or not ibm_password or not ibm_url:
                    return "Укажите IBM Watson Username, Password и URL"
                text = self.recognizer.recognize_ibm(
                    audio_data,
                    username=ibm_username,
                    password=ibm_password,
                    url=ibm_url,
                    language="ru-RU"
                )
            else:
                return "Неизвестный движок"

            return text

        except sr.UnknownValueError:
            return "Речь не распознана"
        except sr.RequestError as e:
            return f"Ошибка сервиса распознавания: {e}"
        except Exception as e:
            return f"Произошла ошибка: {e}"

    def create_interface(self):
        """Создает Gradio интерфейс"""
        with gr.Blocks(title="Преобразование речи в текст") as demo:
            gr.Markdown("# 🎤 Преобразование речи в текст")
            gr.Markdown("Выберите движок, при необходимости укажите ключи, затем запишите аудио или загрузите файл.")

            with gr.Row():
                engine_dropdown = gr.Dropdown(
                    choices=["Google", "Google Cloud", "Sphinx", "Houndify", "IBM Watson"],
                    value="Google",
                    label="Движок распознавания",
                    interactive=True
                )

            with gr.Accordion("🔑 Настройки API (для облачных сервисов)", open=False):
                google_cloud_key = gr.Textbox(
                    label="Google Cloud JSON ключ",
                    placeholder="Вставьте содержимое JSON ключа...",
                    lines=6
                )
                houndify_client_id = gr.Textbox(label="Houndify Client ID")
                houndify_client_key = gr.Textbox(label="Houndify Client Key", type="password")
                ibm_username = gr.Textbox(label="IBM Watson Username")
                ibm_password = gr.Textbox(label="IBM Watson Password", type="password")
                ibm_url = gr.Textbox(label="IBM Watson URL", placeholder="https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/xxx")

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
                recognized_text = gr.Textbox(
                    label="Распознанный текст",
                    placeholder="Здесь появится распознанный текст...",
                    lines=4,
                    max_lines=10,
                    interactive=True,
                    show_copy_button=True
                )

            with gr.Row():
                editing_text = gr.Textbox(
                    label="Редактируемый текст",
                    placeholder="Здесь можно редактировать текст независимо...",
                    lines=6,
                    max_lines=15,
                    interactive=True,
                    show_copy_button=True
                )

            convert_btn.click(
                fn=self.transcribe_audio,
                inputs=[
                    audio_input,
                    engine_dropdown,
                    google_cloud_key,
                    houndify_client_id,
                    houndify_client_key,
                    ibm_username,
                    ibm_password,
                    ibm_url
                ],
                outputs=recognized_text
            )

        return demo

    def launch(self, **kwargs):
        demo = self.create_interface()
        demo.launch(**kwargs)

if __name__ == "__main__":
    app = SpeechToTextApp()
    app.launch(share=True)
