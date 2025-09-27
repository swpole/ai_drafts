import gradio as gr
import whisper

model = whisper.load_model("base")

def transcribe(audio, current_text):
    """Распознаём новый кусок речи и добавляем в поле"""
    if audio is None:
        # сброс без изменений текста
        return current_text, gr.update(value=None, visible=False)
    result = model.transcribe(audio)
    new_text = result["text"].strip()
    combined = (current_text + " " + new_text).strip()
    # очищаем микрофон (value=None), скрываем
    return combined, gr.update(value=None, visible=False)

with gr.Blocks() as demo:
    with gr.Row():
        text = gr.Textbox(label="Сообщение", lines=3)
        mic_btn = gr.Button("🎤", scale=0)

    mic = gr.Microphone(type="filepath", visible=False)

    # Показать микрофон
    mic_btn.click(lambda: gr.update(visible=True), outputs=mic)

    # Обработка аудио после записи
    mic.change(fn=transcribe, inputs=[mic, text], outputs=[text, mic])

demo.launch()
