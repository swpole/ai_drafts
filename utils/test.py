import gradio as gr
from llm_interface import LLMInterface

class SimpleText:
    def __init__(self):
        # Перенесем создание текстового поля в create_interface
        pass
        
    def create_interface(self):
        # Теперь создаем и возвращаем текстовое поле
        self.text_box = gr.Textbox(
            label="Enter some text", 
            placeholder="Type here...", 
            lines=2
        )
        return self.text_box

class Test:
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft(), title="Test Interface") as interface:
            
            t2_component = SimpleText()
            t2_textbox = t2_component.create_interface()

            llm_interface = LLMInterface()
            llm_interface.build_interface()
            
            
        return interface
    
    def launch(self):
        interface = self.create_interface()
        interface.launch()

if __name__ == "__main__":
    Test().launch()