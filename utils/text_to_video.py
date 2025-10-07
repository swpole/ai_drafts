from text_illustrator import TextIllustrator
from video_creator import VideoCreator
import gradio as gr

class TextToVideo:
    def __init__(self, text_prompt: str):
       pass
    
    def create_interface(self):
        
        with gr.Blocks() as iface:
            gr.Markdown("# Текст в Видео 🚀")

            self.illustrator = TextIllustrator()
            self.video_creator = VideoCreator()

            self.illustrator.create_interface()
            self.video_creator.create_interface()

            img_generated=self.illustrator.image_generator.output_image
            img_gallery=self.video_creator.new_img_upload
            img_generated.change(fn=lambda x: x, 
                              inputs=img_generated, 
                              outputs=img_gallery)

            gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")

        return iface

if __name__ == "__main__":
    t2v = TextToVideo("A serene landscape with mountains and a river.")
    iface = t2v.create_interface()
    iface.launch(allowed_paths=t2v.illustrator.allowed_paths)
