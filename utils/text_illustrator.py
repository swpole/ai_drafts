import gradio as gr
from illustration_prompt_generator import IllustrationPromptGenerator
from image_generator_simple_stt import ImageGeneratorSimpleSTT

class TextIllustrator:
    def __init__(self):
        pass

    def create_interface(self):
        # Создаем основной интерфейс
        text_illustrator = "Текст иллюстратор"
        with gr.Blocks(theme=gr.themes.Soft(), title=text_illustrator) as interface:
            title_md =f"# 🚀 {text_illustrator}"
            gr.Markdown(title_md)

            illustration_prompt_generator = IllustrationPromptGenerator()
            prompt_interface, generated_prompt = illustration_prompt_generator.create_interface()
            
            gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")
            
            image_generator = ImageGeneratorSimpleSTT()        
            image_interface,_, pos_prompt = image_generator.create_interface()

            generated_prompt.change(
                fn=lambda x: x,  # Просто передаем значение дальше
                inputs=generated_prompt,
                outputs=pos_prompt
            )

        return interface

if __name__ == "__main__":
    app = TextIllustrator()
    interface = app.create_interface()
    interface.launch(
        #share=True, 
    )