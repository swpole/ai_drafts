import gradio as gr
import whisper

model = whisper.load_model("base")

def insert_at_cursor(audio, text, cursor_pos):
    if audio is None:
        return text, gr.update(value=None, visible=False)

    result = model.transcribe(audio)
    new_text = result["text"].strip()

    if cursor_pos is None:
        cursor_pos = len(text)

    combined = text[:cursor_pos] + new_text + text[cursor_pos:]
    return combined, gr.update(value=None, visible=False)

with gr.Blocks() as demo:
    with gr.Row():
        text = gr.Textbox(label="Сообщение", lines=3, elem_id="my_textbox")
        mic_btn = gr.Button("🎤", scale=0)

    mic = gr.Microphone(type="filepath", visible=False)
    cursor_pos = gr.Number(value=0, visible=False)

    # Показать микрофон
    mic_btn.click(lambda: gr.update(visible=True), outputs=mic)

    # Обработка аудио: вставляем распознанное в позицию курсора
    mic.change(
        fn=insert_at_cursor,
        inputs=[mic, text, cursor_pos],
        outputs=[text, mic]
    )

    # JS: при каждом клике/наборе текста обновляем позицию курсора
    text.change(
        fn=None,
        inputs=None,
        outputs=cursor_pos,
        js="""
        (text) => {
            const box = document.querySelector("#my_textbox textarea");
            return box.selectionStart;
        }
        """
    )

demo.launch()
