import gradio as gr
from deepseek_textbox_with_stt import TextboxWithSTT

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Пример с TextboxWithSTT")
        
        # Создаем наш кастомный Textbox
        custom_textbox = TextboxWithSTT()
        custom_textbox.render()
        
        # Можно использовать как обычный Textbox
        button = gr.Button("Отправить")
        output = gr.Textbox(label="Результат")
        
        button.click(
            fn=lambda x: x,
            inputs=[custom_textbox.component],
            outputs=[output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()