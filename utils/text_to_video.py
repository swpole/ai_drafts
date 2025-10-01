from text_illustrator import TextIllustrator
from video_creator import VideoCreator
import gradio as gr

class TextToVideo:
    def __init__(self, text, image_style='default', video_style='default'):
        self.text = text
        self.image_style = image_style
        self.video_style = video_style
        self.illustrator = TextIllustrator()
        self.video_creator = VideoCreator()
    
    def create_interface(self):
        
        with gr.Blocks() as iface:
            gr.Markdown("# Text to Video Generator")
            
            _, allowed_paths = self.illustrator.create_interface()
            self.video_creator.create_interface()

            img_output=self.illustrator.image_generator.output_image
            img_input=self.video_creator.new_img_upload
            img_output.change(fn=lambda x: x, inputs=img_output, outputs=img_input)

            gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")

        return iface, allowed_paths

if __name__ == "__main__":
    t2v = TextToVideo("A serene landscape with mountains and a river.")
    iface, allowed_paths = t2v.create_interface()
    iface.launch(allowed_paths=allowed_paths)
