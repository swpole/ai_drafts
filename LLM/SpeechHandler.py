import gradio as gr
import whisper

class SpeechHandler:
    """
    Класс для преобразования речи в текст с использованием Whisper
    """
    
    def __init__(self, model_name="large"):
        """
        Инициализация модели Whisper
        
        Args:
            model_name: название модели Whisper (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.whisper_model = None
        self.load_model()
    
    def load_model(self):
        """
        Загрузка модели Whisper для транскрибации
        """
        try:
            self.whisper_model = whisper.load_model(self.model_name)
            print(f"Модель Whisper '{self.model_name}' загружена успешно")
        except Exception as e:
            print(f"Ошибка загрузки Whisper: {e}")
            self.whisper_model = None
    
    def transcribe_audio(self, audio_file_path, language="ru"):
        """
        Преобразует аудиофайл в текст с помощью модели Whisper.
        Возвращает распознанный текст.
        
        Args:
            audio_file_path: путь к аудиофайлу
            language: язык для распознавания (ru, en, etc.)
        """
        if audio_file_path is None:
            return "Аудиофайл не предоставлен."
        
        if self.whisper_model is None:
            return "Модель Whisper не загружена. Проверьте установку."

        print(f"Транскрибация аудио: {audio_file_path}")
        try:
            # Загружаем аудио и применяем модель
            result = self.whisper_model.transcribe(audio_file_path, language=language)
            transcribed_text = result["text"]
            print(f"Распознанный текст: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            return f"Ошибка транскрибации: {str(e)}"
    
    def get_available_models(self):
        """
        Возвращает список доступных моделей Whisper
        """
        return ["tiny", "base", "small", "medium", "large"]
    
    def change_model(self, model_name):
        """
        Смена модели Whisper
        """
        if model_name in self.get_available_models():
            self.model_name = model_name
            self.load_model()
            return f"Модель изменена на: {model_name}"
        else:
            return f"Модель {model_name} недоступна"
    
    def create_interface(self):
        """
        Создает Gradio интерфейс для преобразования речи в текст
        """
        with gr.Blocks(title="Speech to Text", theme="soft") as interface:
            gr.Markdown("# 🎙️ Преобразование речи в текст")

            # Примеры использования
            with gr.Accordion("Инструкция по использованию", open=False):
                gr.Markdown("""
                **Как использовать:**
                1. Выберите подходящую модель Whisper (base - хороший баланс скорости и качества)
                2. Загрузите аудиофайл или запишите голос с микрофона
                3. Нажмите "Преобразовать в текст"
                4. Скопируйте распознанный текст
                
                **Поддерживаемые форматы:** WAV, MP3, FLAC, M4A
                **Рекомендуемая длительность:** до 30 секунд для лучшего качества
                """)

            # Блок для работы с аудио
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Загрузите аудиофайл или запишите с микрофона", 
                     type="filepath", 
                    interactive=True
                )
            
            # Статус модели
            model_status = "🟢 Загружена" if self.whisper_model else "🔴 Не загружена"
            gr.Markdown(f"**Статус модели Whisper:** {model_status} ({self.model_name})")

            with gr.Row():
                # Выбор модели Whisper
                with gr.Column(scale=2):
                    model_selector = gr.Dropdown(
                        choices=self.get_available_models(),
                        value=self.model_name,
                        label="Выберите модель Whisper",
                        info="Большие модели точнее, но медленнее"
                    )
                with gr.Column(scale=1):
                    change_model_btn = gr.Button("🔄 Сменить модель")
                with gr.Column(scale=1):
                    model_status_output = gr.Textbox(label="Статус смены модели", interactive=False)

            with gr.Row():
                # Блок для работы с аудио
                with gr.Column(scale=1):
                    language_selector = gr.Dropdown(
                            choices=["ru", "en", "auto"],
                            value="ru",
                            label="Язык распознавания"
                    )
                    
                    transcribe_button = gr.Button("🔊 Преобразовать в текст", variant="primary")

                # Блок для вывода текста
                with gr.Column(scale=2):
                    text_output = gr.Textbox(
                        label="Распознанный текст", 
                        lines=8, 
                        placeholder="Здесь появится распознанный текст...",
                        interactive=True,
                        show_copy_button=True
                    )
            
            # Обработчики событий
            
            # Транскрибация аудио
            transcribe_button.click(
                fn=self.transcribe_audio, 
                inputs=[audio_input, language_selector], 
                outputs=text_output
            )
            
            # Смена модели
            change_model_btn.click(
                fn=self.change_model,
                inputs=model_selector,
                outputs=model_status_output
            )

        return interface

    def launch_interface(self, server_name="0.0.0.0", share=False, debug=True, port=7860):

        interface = self.create_interface()
        
        print(f"Запуск интерфейса SpeechHandler на порту {port}...")
        print(f"Откройте http://localhost:{port} в вашем браузере")
        
        try:
            interface.launch(
                server_name=server_name,
                share=share,
                debug=debug,
                server_port=port
            )
        except Exception as e:
            print(f"Ошибка запуска интерфейса: {e}")
            print(f"Проверьте, что порт {port} свободен или укажите другой порт")

if __name__ == "__main__":
    # Тестирование класса SpeechHandler
    speech_handler = SpeechHandler("large")
    
    print("=== Тестирование SpeechHandler ===")
    print(f"Доступные модели: {speech_handler.get_available_models()}")
    print(f"Текущая модель: {speech_handler.model_name}")
    print(f"Модель загружена: {speech_handler.whisper_model is not None}")
    
    # Запускаем интерфейс
    speech_handler.launch_interface(port=7860)