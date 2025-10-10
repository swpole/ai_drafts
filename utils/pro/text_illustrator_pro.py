#input - text
#output - image

import gradio as gr
from illustration_prompt_generator_pro import IllustrationPromptGenerator
from image_generator_simple_stt_pro import ImageGeneratorSimpleSTT

class TextIllustratorPro:
    def __init__(self):
        self.create_interface()
        pass

    def create_interface(self):
        # Создаем основной интерфейс
        text_illustrator = "Текст иллюстратор"
        
        title_md =f"# 🚀 {text_illustrator}"
        gr.Markdown(title_md)

        self.illustration_prompt_generator = IllustrationPromptGenerator()
        
        gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")
        
        self.image_generator = ImageGeneratorSimpleSTT()            
        self.allowed_paths = [self.image_generator.output_dir]

        self.illustration_prompt_generator.scene_prompt_box.change(
            fn=lambda x: x, 
            inputs=self.illustration_prompt_generator.scene_prompt_box, 
            outputs=self.image_generator.positive_prompt.textbox
        ) 

        return

if __name__ == "__main__":

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        app = TextIllustratorPro()    
        
    interface.launch(
        allowed_paths=app.allowed_paths,
        #share=True, 
    )