#input - text
#output - video

from text_illustrator_online import TextIllustratorOnline
from video_creator_offline import VideoCreatorOffline
import gradio as gr

class TextToVideoOnline:
    def __init__(self):
       self.create_interface()
       pass
    
    def create_interface(self):
        

        gr.Markdown("# Текст в Видео 🚀")

        self.illustrator = TextIllustratorOnline()
        self.video_creator = VideoCreatorOffline()

        img_generated=self.illustrator.image_generator.output_image
        img_gallery=self.video_creator.new_img_upload
        img_generated.change(fn=lambda x: x, 
                            inputs=img_generated, 
                            outputs=img_gallery)

        gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")

        return

if __name__ == "__main__":
    with gr.Blocks() as iface:
        t2v = TextToVideoOnline()

    iface.launch()
