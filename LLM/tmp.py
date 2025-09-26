import gradio as gr

# глобальная переменная
shared_text = ""

def set_text_from_speech(text):
    global shared_text
    shared_text = text
    return f"Speech text saved: {text}"

def get_text_for_llm():
    global shared_text
    return shared_text

with gr.Blocks() as speech:
    inp = gr.Textbox(label="Speech input")
    out = gr.Textbox(label="Status")
    btn = gr.Button("Save text")
    btn.click(set_text_from_speech, inp, out)

with gr.Blocks() as llm:
    out = gr.Textbox(label="LLM sees this")
    btn = gr.Button("Load text from speech")
    btn.click(get_text_for_llm, None, out)

demo = gr.TabbedInterface([speech, llm], ["Speech", "LLM"])
demo.launch()
