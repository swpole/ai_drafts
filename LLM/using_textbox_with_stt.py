# main.py
import gradio as gr
from textbox_with_stt_final import TextboxWithSTT

# создаём экземпляры компонента (можно несколько)
stt1 = TextboxWithSTT()
stt2 = TextboxWithSTT()

with gr.Blocks() as demo:
    gr.Markdown("## Пример использования TextboxWithSTT из другого файла")

    with gr.Row():
        txt1 = stt1.render()
        txt2 = stt2.render()

    out1 = gr.Textbox(label="Вывод 1")
    out2 = gr.Textbox(label="Вывод 2")

    btn1 = gr.Button("Показать текст из первого")
    btn2 = gr.Button("Показать текст из второго")

    btn1.click(fn=lambda x: x, inputs=txt1, outputs=out1)
    btn2.click(fn=lambda x: x, inputs=txt2, outputs=out2)

demo.launch()

