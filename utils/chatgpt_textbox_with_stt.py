"""
TextboxWithSTT — расширяемый Gradio-компонент для ввода текста + голосовое распознавание
"""

from typing import Dict, Any, Optional, Tuple

try:
    import gradio as gr
except Exception as e:
    raise ImportError("Gradio не установлен. Установите gradio>=3.0") from e

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import whisper
except Exception:
    whisper = None


class TextboxWithSTT:
    DEFAULT_SR_ENGINES = ["google", "sphinx", "wit"]
    DEFAULT_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, textbox_kwargs: Optional[Dict[str, Any]] = None):
        self.textbox_kwargs = textbox_kwargs or {}
        self._components = {}

    def render(self, root: Optional[gr.Blocks] = None):
        should_close = False
        if root is None:
            root = gr.Blocks()
            should_close = True

        with root:
            cursor_pos = gr.Number(value=0, label="Cursor position", visible=False)

            with gr.Accordion("Голосовой ввод", open=False):
                engine_select = gr.Radio(choices=["speech_recognition", "whisper"], value="speech_recognition", label="Движок распознавания")
                with gr.Accordion("Настройки API", open=False):
                    api_settings = gr.Textbox(label="Настройки API (JSON или текст)", placeholder='{"google_api_key": "..."}', lines=2)
                with gr.Accordion("Аудиовход", open=False):
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Аудиовход (микрофон или файл)")
                sr_engine_dropdown = gr.Dropdown(choices=self.DEFAULT_SR_ENGINES, value=self.DEFAULT_SR_ENGINES[0], label="SR: движок (если выбрано speech_recognition)")
                whisper_model_dropdown = gr.Dropdown(choices=self.DEFAULT_WHISPER_MODELS, value="small", label="Whisper: модель")
                lang_select = gr.Textbox(label="Язык распознавания (например, 'en-US' или 'ru-RU')", value="ru-RU")

            elem_id = self.textbox_kwargs.get("elem_id", "textbox_with_stt_main")
            textbox = gr.Textbox(**self.textbox_kwargs, elem_id=elem_id, lines=5)

            self._components.update({
                "textbox": textbox,
                "cursor_pos": cursor_pos,
                "engine_select": engine_select,
                "api_settings": api_settings,
                "audio_in": audio_in,
                "sr_engine_dropdown": sr_engine_dropdown,
                "whisper_model_dropdown": whisper_model_dropdown,
                "lang_select": lang_select,
            })

            get_cursor_js = f"""
                () => {{
                    const box = document.querySelector("#{elem_id} textarea");
                    return box ? box.selectionStart : 0;
                }}
            """

            self.textbox = textbox
            self.textbox.select(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
            textbox.select(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
            textbox.input(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
            textbox.focus(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
            textbox.blur(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)
            textbox.submit(fn=None, inputs=None, outputs=cursor_pos, js=get_cursor_js)

            audio_in.change(
                fn=self._transcribe_and_insert,
                inputs=[textbox, cursor_pos, engine_select, sr_engine_dropdown, whisper_model_dropdown, api_settings, audio_in, lang_select],
                outputs=[textbox, cursor_pos],
            )

        if should_close:
            return root
        return self._components

    def _insert_at_cursor(self, current_text: str, insert_text: str, cursor_pos: int) -> Tuple[str, int]:
        if current_text is None:
            current_text = ""
        if cursor_pos is None or cursor_pos < 0:
            new_text = current_text + insert_text
            return new_text, len(new_text)
        cursor_pos = min(max(0, int(cursor_pos)), len(current_text))
        new_text = current_text[:cursor_pos] + insert_text + current_text[cursor_pos:]
        return new_text, cursor_pos + len(insert_text)

    def _transcribe_and_insert(self, textbox_value: str, cursor_pos: int, engine_choice: str, sr_engine: str, whisper_model: str, api_settings_str: str, audio_path: Optional[str], lang: str):
        if not audio_path:
            return gr.update(value=textbox_value), cursor_pos
        try:
            if engine_choice == "speech_recognition":
                if sr is None:
                    raise RuntimeError("Модуль speech_recognition не установлен")
                transcript = self._transcribe_with_speech_recognition(audio_path, sr_engine, api_settings_str, lang)
            elif engine_choice == "whisper":
                if whisper is None:
                    raise RuntimeError("Whisper не установлен")
                transcript = self._transcribe_with_whisper(audio_path, api_settings_str, lang, whisper_model)
            else:
                transcript = ""
        except Exception as e:
            err = f"[ERR STT] {str(e)}"
            return self._insert_at_cursor(textbox_value or "", err, cursor_pos)
        return self._insert_at_cursor(textbox_value or "", transcript, cursor_pos)

    def _transcribe_with_speech_recognition(self, audio_path: str, engine: str, api_settings_str: str, lang: str) -> str:
        if sr is None:
            raise RuntimeError("speech_recognition не доступен")
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        engine = engine.lower()
        if engine == "google":
            return r.recognize_google(audio_data, language=lang or "en-US")
        elif engine == "sphinx":
            return r.recognize_sphinx(audio_data, language=lang or "en-US")
        elif engine == "wit":
            token = None
            if api_settings_str:
                for part in api_settings_str.replace('\n', ' ').split():
                    if 'WIT' in part.upper():
                        token = part.split('=')[-1].strip()
            if not token:
                raise RuntimeError('Wit token not found')
            return r.recognize_wit(audio_data, key=token)
        else:
            raise RuntimeError(f"Неизвестный engine: {engine}")

    def _transcribe_with_whisper(self, audio_path: str, api_settings_str: str, lang: str, model_name: str) -> str:
        if not model_name:
            model_name = "small"
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, language=lang if lang else None)
        return result.get("text", "")


if __name__ == "__main__":
    stt = TextboxWithSTT(textbox_kwargs={"label": "Текст", "placeholder": "Печатайте или используйте голосовой ввод"})
    app = stt.render()
    if isinstance(app, gr.Blocks):
        app.launch()
