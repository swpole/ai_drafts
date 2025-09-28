import gradio as gr
import speech_recognition as sr
import numpy as np
import tempfile
import wave
import uuid
from typing import Union, Tuple, Optional


class TextboxWithSTT:
    """
    Компонент Gradio: текстовое поле с голосовым вводом (STT).
    Создавайте экземпляры и вставляйте в gr.Blocks() через .render().
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.elem_id = f"stt_textbox_{uuid.uuid4().hex[:8]}"
        self.textbox: Optional[gr.Textbox] = None

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
            # Если путь к файлу
            if isinstance(audio, str):
                with sr.AudioFile(audio) as source:
                    audio_data = self.recognizer.record(source)
            else:  # (sr, np.ndarray)
                sr_rate, y = audio
                if y.ndim > 1:
                    y = y.mean(axis=1)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    with wave.open(tmp.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr_rate)
                        wf.writeframes((y * 32767).astype(np.int16).tobytes())
                    with sr.AudioFile(tmp.name) as source:
                        audio_data = self.recognizer.record(source)

            # Выбор движка
            if engine == "Google":
                return self.recognizer.recognize_google(audio_data, language="ru-RU")

            if engine == "Google Cloud":
                if not google_cloud_key:
                    return "Укажите Google Cloud JSON ключ"
                return self.recognizer.recognize_google_cloud(
                    audio_data, credentials_json=google_cloud_key, language="ru-RU"
                )

            if engine == "Sphinx":
                return self.recognizer.recognize_sphinx(audio_data, language="ru-RU")

            if engine == "Houndify":
                if not houndify_client_id or not houndify_client_key:
                    return "Укажите Houndify Client ID и Client Key"
                return self.recognizer.recognize_houndify(
                    audio_data, client_id=houndify_client_id, client_key=houndify_client_key
                )

            if engine == "IBM Watson":
                if not ibm_username or not ibm_password or not ibm_url:
                    return "Укажите IBM Watson Username, Password и URL"
                return self.recognizer.recognize_ibm(
                    audio_data,
                    username=ibm_username,
                    password=ibm_password,
                    url=ibm_url,
                    language="ru-RU"
                )

            return "Неизвестный движок"

        except sr.UnknownValueError:
            return "Речь не распознана"
        except sr.RequestError as e:
            return f"Ошибка сервиса распознавания: {e}"
        except Exception as e:
            return f"Произошла ошибка: {e}"

    def insert_at_cursor(
        self,
        audio_file,
        text,
        cursor_pos,
        engine: str,
        google_cloud_key: str,
        houndify_client_id: str,
        houndify_client_key: str,
        ibm_username: str,
        ibm_password: str,
        ibm_url: str
    ):
        if audio_file is None:
            return text, gr.update(value=None)

        result = self.transcribe_audio(
            audio_file,
            engine,
            google_cloud_key,
            houndify_client_id,
            houndify_client_key,
            ibm_username,
            ibm_password,
            ibm_url
        )

        if cursor_pos is None:
            cursor_pos = len(text or "")
        base_text = text or ""
        combined = base_text[:int(cursor_pos)] + result + base_text[int(cursor_pos):]

        return combined, gr.update(value=None)

    def render(self) -> gr.Textbox:
        """
        Создаёт компоненты внутри текущего gr.Blocks() и возвращает gr.Textbox.
        """
        with gr.Accordion("🎤 Голосовой ввод", open=False):
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
                ibm_url = gr.Textbox(
                    label="IBM Watson URL",
                    placeholder="https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/xxx"
                )

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Аудиовход",
                interactive=True
            )

            cursor_pos = gr.Number(value=0, visible=False)

        self.textbox = gr.Textbox(
            placeholder="Поставьте курсор в нужное место и добавьте аудио...",
            lines=4,
            max_lines=10,
            interactive=True,
            show_copy_button=True,
            elem_id=self.elem_id
        )

        # события
        audio_input.change(
            fn=self.insert_at_cursor,
            inputs=[
                audio_input, self.textbox, cursor_pos,
                engine_dropdown,
                google_cloud_key,
                houndify_client_id,
                houndify_client_key,
                ibm_username,
                ibm_password,
                ibm_url
            ],
            outputs=[self.textbox, audio_input]
        )

        # JS для курсора
        get_cursor_js = f"""
            () => {{
                const box = document.querySelector("#{self.elem_id} textarea");
                return box ? box.selectionStart : 0;
            }}
        """
        self.textbox.input(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.change(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.submit(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.focus(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.blur(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.select(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)

        return self.textbox

    def launch(self, **kwargs):
        """Запуск отдельного интерфейса"""
        with gr.Blocks() as demo:
            self.render()
        demo.launch(**kwargs)

if __name__ == "__main__":
    app = TextboxWithSTT()
    app.launch(share=False)