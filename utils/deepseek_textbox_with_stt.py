import gradio as gr
import speech_recognition as sr
import whisper
from typing import Optional, Dict, Any
import json
import uuid

class TextboxWithSTT:
    def __init__(self, **kwargs):
        """
        Инициализация TextboxWithSTT
        
        Args:
            **kwargs: Аргументы для gr.Textbox
        """
        # уникальный elem_id для textarea (чтобы можно было создать несколько экземпляров)
        self.elem_id = f"stt_textbox_{uuid.uuid4().hex[:8]}"

        self.textbox: Optional[gr.Textbox] = None
        self.audio_input = None
        self.engine_dropdown = None
        self.free_engine_dropdown = None
        self.api_key_textbox = None
        self.recognition_result = None
        self.current_cursor_pos = 0
        
    def get_js_function(self):
        """Возвращает JS функцию для отслеживания позиции курсора"""
        return """
        function() {
            const textbox = document.activeElement;
            if (textbox && textbox.type === 'textarea') {
                return {
                    cursor_pos: textbox.selectionStart,
                    text: textbox.value,
                    selection_start: textbox.selectionStart,
                    selection_end: textbox.selectionEnd
                };
            }
            return {cursor_pos: 0, text: '', selection_start: 0, selection_end: 0};
        }
        """
    
    def update_cursor_position(self, data: str) -> int:
        """
        Обновляет позицию курсора из JS данных
        
        Args:
            data: JSON строка с данными о позиции курсора
            
        Returns:
            Текущая позиция курсора
        """
        if data:
            try:
                cursor_data = json.loads(data)
                self.current_cursor_pos = cursor_data.get('cursor_pos', 0)
                return self.current_cursor_pos
            except:
                pass
        return self.current_cursor_pos
    
    def insert_text_at_cursor(self, current_text: str, new_text: str, cursor_pos: int) -> str:
        """
        Вставляет текст в текущую позицию курсора
        
        Args:
            current_text: Текущий текст
            new_text: Текст для вставки
            cursor_pos: Позиция курсора
            
        Returns:
            Обновленный текст
        """
        if not new_text:
            return current_text
            
        # Вставляем текст в позицию курсора
        if cursor_pos >= len(current_text):
            return current_text + new_text
        else:
            return current_text[:cursor_pos] + new_text + current_text[cursor_pos:]
    
    def recognize_speech(self, audio_data: tuple, engine: str, free_engine: str, 
                        api_key: str, current_text: str, cursor_data: str) -> str:
        """
        Распознает речь и вставляет текст в позицию курсора
        
        Args:
            audio_data: Данные аудио из gr.Audio
            engine: Выбранный движок ('speech_recognition' или 'whisper')
            free_engine: Бесплатный движок для speech_recognition
            api_key: API ключ
            current_text: Текущий текст
            cursor_data: Данные о позиции курсора
            
        Returns:
            Обновленный текст
        """
        if audio_data is None:
            return current_text
            
        # Обновляем позицию курсора
        cursor_pos = self.update_cursor_position(cursor_data)
        
        try:
            recognized_text = ""
            
            if engine == "speech_recognition":
                recognized_text = self._recognize_with_speech_recognition(
                    audio_data, free_engine, api_key
                )
            elif engine == "whisper":
                recognized_text = self._recognize_with_whisper(audio_data)
            
            # Вставляем распознанный текст в позицию курсора
            if recognized_text:
                return self.insert_text_at_cursor(current_text, recognized_text, cursor_pos)
                
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            
        return current_text
    
    def _recognize_with_speech_recognition(self, audio_data: tuple, 
                                         engine: str, api_key: str) -> str:
        """
        Распознавание с использованием speech_recognition
        
        Args:
            audio_data: Данные аудио
            engine: Выбранный движок
            api_key: API ключ
            
        Returns:
            Распознанный текст
        """
        recognizer = sr.Recognizer()
        sr_audio_data = sr.AudioData(
            audio_data[1].tobytes(), 
            audio_data[0], 
            audio_data[1].itemsize
        )
        
        try:
            if engine == "google":
                return recognizer.recognize_google(sr_audio_data, language="ru-RU")
            elif engine == "google_cloud" and api_key:
                return recognizer.recognize_google_cloud(sr_audio_data, credentials_json=api_key, language="ru-RU")
            elif engine == "wit" and api_key:
                return recognizer.recognize_wit(sr_audio_data, key=api_key)
            elif engine == "bing" and api_key:
                return recognizer.recognize_bing(sr_audio_data, key=api_key)
            elif engine == "houndify" and api_key:
                return recognizer.recognize_houndify(sr_audio_data, client_id=api_key.split('|')[0], client_key=api_key.split('|')[1] if '|' in api_key else "")
            elif engine == "ibm" and api_key:
                username = api_key.split('|')[0]
                password = api_key.split('|')[1] if '|' in api_key else ""
                return recognizer.recognize_ibm(sr_audio_data, username=username, password=password, language="ru-RU")
            else:
                # По умолчанию используем Google
                return recognizer.recognize_google(sr_audio_data, language="ru-RU")
                
        except sr.UnknownValueError:
            return "Речь не распознана"
        except sr.RequestError as e:
            return f"Ошибка сервиса: {e}"
    
    def _recognize_with_whisper(self, audio_data: tuple) -> str:
        """
        Распознавание с использованием Whisper
        
        Args:
            audio_data: Данные аудио
            
        Returns:
            Распознанный текст
        """
        try:
            # Сохраняем временный файл для Whisper
            import tempfile
            import wave
            import numpy as np
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)  # моно
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(audio_data[0])
                    wf.writeframes(audio_data[1].tobytes())
                
                # Загружаем модель Whisper (базовая модель)
                model = whisper.load_model("base")
                result = model.transcribe(temp_file.name, language="ru")
                return result["text"]
                
        except Exception as e:
            return f"Ошибка Whisper: {e}"
    
    def render(self, **textbox_kwargs):
        """Рендерит все компоненты"""
        with gr.Accordion("Голосовой ввод", open=False):
            with gr.Accordion("Движок распознавания", open=False):
                self.engine_dropdown = gr.Dropdown(
                    choices=["speech_recognition", "whisper"],
                    value="speech_recognition",
                    label="Выберите движок распознавания"
                )
                
                self.free_engine_dropdown = gr.Dropdown(
                    choices=[
                        "google", 
                        "google_cloud", 
                        "wit", 
                        "bing", 
                        "houndify", 
                        "ibm"
                    ],
                    value="google",
                    label="Выберите движок speech_recognition",
                    visible=True
                )
            
            with gr.Accordion("Настройки API", open=False):
                self.api_key_textbox = gr.Textbox(
                    label="API ключ (если требуется)",
                    type="password",
                    placeholder="Введите ваш API ключ...",
                    visible=True
                )
            
            self.audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Аудиовход"
            )
            
            self.recognition_result = gr.Textbox(
                label="Результат распознавания",
                interactive=False,
                visible=False
            )
        
        # Скрываем/показываем настройки в зависимости от выбранного движка
        def update_engine_visibility(engine):
            if engine == "speech_recognition":
                return {
                    self.free_engine_dropdown: gr.update(visible=True),
                    self.api_key_textbox: gr.update(visible=True)
                }
            else:
                return {
                    self.free_engine_dropdown: gr.update(visible=False),
                    self.api_key_textbox: gr.update(visible=False)
                }
        
        self.engine_dropdown.change(
            update_engine_visibility,
            inputs=[self.engine_dropdown],
            outputs=[self.free_engine_dropdown, self.api_key_textbox]
        )
        
        # События для отслеживания позиции курсора
        cursor_tracker = gr.JSON(visible=True)

        self.textbox = gr.Textbox(
            elem_id=self.elem_id,
            **textbox_kwargs,  # <--- сюда можно кидать label, placeholder, lines, value
        )

        # JS для получения позиции курсора по уникальному elem_id
        get_cursor_js = f"""
            () => {{
                const box = document.querySelector("#{self.elem_id} textarea");
                return box ? box.selectionStart : 0;
            }}
        """
        
        # Привязываем события Textbox к JS функции
        self.textbox.change(
            fn=lambda x: x,
            inputs=[self.textbox],
            outputs=[cursor_tracker],
            js=self.get_js_function()
        )
        
        self.textbox.input(
            fn=lambda x: x,
            inputs=[self.textbox],
            outputs=[cursor_tracker],
            js=self.get_js_function()
        )
        
        self.textbox.select(
            fn=lambda x: x,
            inputs=[self.textbox],
            outputs=[cursor_tracker],
            js=self.get_js_function()
        )
        
        # Обработка аудио ввода
        self.audio_input.stop_recording(
            fn=self.recognize_speech,
            inputs=[
                self.audio_input,
                self.engine_dropdown,
                self.free_engine_dropdown,
                self.api_key_textbox,
                self.textbox,
                cursor_tracker
            ],
            outputs=[self.textbox]
        )
        
        self.audio_input.upload(
            fn=self.recognize_speech,
            inputs=[
                self.audio_input,
                self.engine_dropdown,
                self.free_engine_dropdown,
                self.api_key_textbox,
                self.textbox,
                cursor_tracker
            ],
            outputs=[self.textbox]
        )
    
    def __getattr__(self, name):
        """Делегирует доступ к атрибутам Textbox"""
        return getattr(self.textbox, name)
    
    @property
    def component(self):
        """Возвращает Textbox компонент для использования в интерфейсе"""
        return self.textbox