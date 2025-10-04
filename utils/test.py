import gradio as gr
from illustration_prompt_generator import IllustrationPromptGenerator
from image_generator_simple_stt import ImageGeneratorSimpleSTT

class Test:
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft(), title="Test Interface") as interface:

            title_md = "# 🚀 Test Interface"
            gr.Markdown(title_md)

            illustration_prompt_generator = IllustrationPromptGenerator()
            illustration_prompt_generator.create_interface()
            generated_prompt = illustration_prompt_generator.scene_prompt_box
            
            gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")
            
            image_generator = ImageGeneratorSimpleSTT()        
            image_generator.create_interface()
            pos_prompt = image_generator.positive_prompt

            generated_prompt.change(
                fn=lambda x: x,  # Просто передаем значение дальше
                inputs=generated_prompt,
                outputs=pos_prompt
            )
            
            
        return interface
    
    def launch(self):
        interface = self.create_interface()
        interface.launch()

if __name__ == "__main__":
    Test().launch()