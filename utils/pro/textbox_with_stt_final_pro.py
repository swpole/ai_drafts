# textbox with speech to text

import gradio as gr
import speech_recognition as sr
import numpy as np
import tempfile
import wave
import uuid
import torch
import whisper                       # OpenAI Whisper
from faster_whisper import WhisperModel  # Faster-Whisper
from typing import Union, Tuple, Optional


class TextboxWithSTTPro:
    """
    Компонент Gradio: текстовое поле + голосовой ввод (несколько движков).
    Создавайте экземпляры и встраивайте их в gr.Blocks() через stt.render().
    """

    def __init__(self, **textbox_kwargs):
        self.recognizer = sr.Recognizer()

        # Кэш моделей, чтобы не грузить заново
        #self.whisper_models = {}
        #self.faster_whisper_models = {}

        self.elem_id = f"stt_textbox_{uuid.uuid4().hex[:8]}"
        self.textbox: Optional[gr.Textbox] = None
        self.render(**textbox_kwargs)

    def _save_temp_wav(self, sr_rate, y) -> str:
        """Сохраняет np.ndarray в temp wav и возвращает путь."""
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr_rate)
            wf.writeframes((y * 32767).astype(np.int16).tobytes())
        return tmp.name
    
    def get_whisper_model(self, model_size: str):
        """Загружает и кэширует openai/whisper модель"""
        if model_size not in self.whisper_models:
            print(f"[INFO] Загружаю Whisper ({model_size})...")
            self.whisper_models[model_size] = whisper.load_model(model_size)
        return self.whisper_models[model_size]

    def get_faster_whisper_model(self, model_size: str):
        """Загружает и кэширует faster-whisper модель"""
        if model_size not in self.faster_whisper_models:
            print(f"[INFO] Загружаю Faster-Whisper ({model_size}) на {self.fw_device}...")
            try:
                model = WhisperModel(model_size, device=self.fw_device, compute_type=self.fw_compute_type)
            except ValueError:
                # fallback если float16 не поддерживается
                model = WhisperModel(model_size, device=self.fw_device, compute_type="float32")
            self.faster_whisper_models[model_size] = model
        return self.faster_whisper_models[model_size]    

    def transcribe_audio(
        self,
        audio: Optional[Union[str, Tuple[int, np.ndarray]]],
        engine: str,
        whisper_model_size: str,
        faster_whisper_model_size: str,
        google_cloud_key: str,
        houndify_client_id: str,
        houndify_client_key: str,
        ibm_username: str,
        ibm_password: str,
        ibm_url: str,
    ) -> str:
        """Преобразует аудио в текст через выбранный движок."""
        if audio is None:
            return "Аудио не предоставлено"

        try:
            # Whisper/OpenAI
            if engine == "Whisper":
                model = self.get_whisper_model(whisper_model_size)
                file_path = audio if isinstance(audio, str) else self._save_temp_wav(*audio)
                result = model.transcribe(file_path, language="ru")
                return result["text"].strip()

            # Faster-Whisper
            if engine == "Faster-Whisper":
                model = self.get_faster_whisper_model(faster_whisper_model_size)
                file_path = audio if isinstance(audio, str) else self._save_temp_wav(*audio)
                segments, _ = model.transcribe(file_path, language="ru")
                return " ".join([seg.text for seg in segments]).strip()

            # Остальные движки через speech_recognition
            if isinstance(audio, str):
                with sr.AudioFile(audio) as source:
                    audio_data = self.recognizer.record(source)
            else:
                sr_rate, y = audio
                file_path = self._save_temp_wav(sr_rate, y)
                with sr.AudioFile(file_path) as source:
                    audio_data = self.recognizer.record(source)

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
                    language="ru-RU",
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
        whisper_model_size: str,
        faster_whisper_model_size: str,
        google_cloud_key: str,
        houndify_client_id: str,
        houndify_client_key: str,
        ibm_username: str,
        ibm_password: str,
        ibm_url: str,
    ):
        """Вставка распознанного текста в позицию курсора."""
        if audio_file is None:
            return text, gr.update(value=None)

        result_text = self.transcribe_audio(
            audio_file,
            engine,
            whisper_model_size,
            faster_whisper_model_size,
            google_cloud_key,
            houndify_client_id,
            houndify_client_key,
            ibm_username,
            ibm_password,
            ibm_url,
        )

        if cursor_pos is None:
            cursor_pos = len(text or "")

        base_text = text or ""
        combined = base_text[:int(cursor_pos)] + result_text + base_text[int(cursor_pos):]

        return combined, gr.update(value=None)

    def render(self, **textbox_kwargs) -> gr.Textbox:
        """Создаёт UI."""
        with gr.Column():
            with gr.Accordion("🎤 Голосовой ввод", open=False):
                engine_dropdown = gr.Dropdown(
                    choices=[
                        "Google", "Google Cloud", "Sphinx", "Houndify", "IBM Watson",
                        "Whisper", "Faster-Whisper"
                    ],
                    value="Google",
                    label="Движок распознавания",
                    interactive=True,
                )

                whisper_model_dropdown = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="Whisper модель",
                )

                faster_whisper_model_dropdown = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="Faster-Whisper модель",
                )

                with gr.Accordion("🔑 Настройки API (для облачных сервисов)", open=False):
                    google_cloud_key = gr.Textbox(
                        label="Google Cloud JSON ключ",
                        placeholder="Вставьте содержимое JSON ключа...",
                        lines=6,
                    )
                    houndify_client_id = gr.Textbox(label="Houndify Client ID")
                    houndify_client_key = gr.Textbox(label="Houndify Client Key", type="password")
                    ibm_username = gr.Textbox(label="IBM Watson Username")
                    ibm_password = gr.Textbox(label="IBM Watson Password", type="password")
                    ibm_url = gr.Textbox(
                        label="IBM Watson URL",
                        placeholder="https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/xxx",
                    )

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Аудиовход",
                    interactive=True,
                )

                cursor_pos = gr.Number(value=0, visible=False)

        self.textbox = gr.Textbox(
            elem_id=self.elem_id,
            interactive=True,
            show_copy_button=True,
            **textbox_kwargs,
        )

        audio_input.change(
            fn=self.insert_at_cursor,
            inputs=[
                audio_input,
                self.textbox,
                cursor_pos,
                engine_dropdown,
                whisper_model_dropdown,
                faster_whisper_model_dropdown,
                google_cloud_key,
                houndify_client_id,
                houndify_client_key,
                ibm_username,
                ibm_password,
                ibm_url,
            ],
            outputs=[self.textbox, audio_input],
        )

        get_cursor_js = f"""
            () => {{
                const box = document.querySelector("#{self.elem_id} textarea");
                return box ? box.selectionStart : 0;
            }}
        """

        self.textbox.change(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.input(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.submit(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.focus(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.blur(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
        self.textbox.select(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)

        return self


if __name__ == "__main__":
    with gr.Blocks() as demo:
        stt = TextboxWithSTTPro(
            label="Системный промпт (роль модели)",
            placeholder="Например: Ты - опытный политолог...",
            lines=3,
            value="Ты - полезный AI ассистент. Отвечай точно и информативно."
        )
    demo.launch(share=False)
