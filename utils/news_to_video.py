import gradio as gr
from news_summarizer import NewsSummarizer
from text_to_video import TextToVideo

class NewsToVideo:
    def __init__(self):
        pass


    def create_interface(self):
        with gr.Blocks() as interface:
            self.summarizer = NewsSummarizer()
            self.summarizer.create_interface()

            self.text_to_video = TextToVideo("")
            self.text_to_video.create_interface()

            self.summarizer.summary_output.change(fn=lambda x: x,
                                                  inputs=self.summarizer.summary_output,
                                                  outputs=self.text_to_video.illustrator.illustration_prompt_generator.input_box)

        return interface

if __name__ == "__main__":
    app = NewsToVideo()
    interface = app.create_interface()
    interface.launch()