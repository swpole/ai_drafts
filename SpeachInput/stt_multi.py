import gradio as gr
import os

def transcribe_audio(audio_path):
    # Заглушка для функции транскрипции аудио в текст (например, с помощью Whisper)
    # В реальности здесь может быть вызов модели speech-to-text
    return "[Транскрибированный текст из аудио]"

def user_message_handler(message, history, audio_files):
    # message - текст пользователя
    # audio_files - список загруженных/записанных аудиофайлов
    new_history = history.copy()

    # 1. Добавляем аудиофайлы в историю как сообщения пользователя
    for audio_file in audio_files:
        if audio_file:  # Если файл есть
            # Для отображения аудио в чате используется специальный формат
            transcribed_text = transcribe_audio(audio_file)  # Транскрибируем аудио
            new_history.append({"role": "user", "content": f"Аудио: {transcribed_text}"})

    # 2. Добавляем текстовое сообщение пользователя, если оно есть
    if message.strip():
        new_history.append({"role": "user", "content": message})

    return "", new_history, []  # Очищаем текстовое поле и список файлов

def bot_response(history):
    # Простейшая логика бота: отвечает на последнее сообщение
    last_user_message = history[-1]["content"]
    bot_reply = f"Вы сказали: {last_user_message}"
    history.append({"role": "assistant", "content": bot_reply})
    return history

with gr.Blocks() as demo:
    gr.Markdown("# Чат-бот с аудиовводом")
    chatbot = gr.Chatbot(type="messages", height=500)
    with gr.Row():
        # Текстовый ввод
        text_input = gr.Textbox(placeholder="Введите текст...", scale=4)
        # Аудиоввод с микрофона
        audio_input = gr.Audio(sources=["microphone"], type="filepath", interactive=True, scale=1)
        # Кнопка загрузки аудиофайла
        file_upload = gr.UploadButton(label="📁", file_types=[".wav", ".mp3"], scale=1)

    clear_btn = gr.Button("Очистить")

    # Обработка событий
    # Объединяем все inputs (текст, аудио с микрофона, загруженные файлы) в одном обработчике
    inputs = [text_input, audio_input, file_upload]
    submit_event = text_input.submit(
        fn=user_message_handler,
        inputs=[text_input, chatbot, file_upload],
        outputs=[text_input, chatbot, file_upload],
        queue=False
    ).then(
        fn=bot_response,
        inputs=chatbot,
        outputs=chatbot
    )

    clear_btn.click(lambda: (None, [], None), None, [text_input, chatbot, file_upload])

demo.launch()