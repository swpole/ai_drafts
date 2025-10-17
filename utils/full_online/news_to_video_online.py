import gradio as gr
from news_summarizer_online import NewsSummarizerOnline
from text_to_audio_online import TextToAudioOnline
from text_to_video_online import TextToVideoOnline

class NewsToVideoPro:

    def __init__(self):
        self.create_interface()

    def create_interface(self):

        news_summerizer = NewsSummarizerOnline()

        text_to_audio = TextToAudioOnline()

        text_to_video = TextToVideoOnline()    

        news_summerizer.llm_interface.output_box.textbox.change(
            fn = lambda x: [x,x,x,x],
            inputs=news_summerizer.llm_interface.output_box.textbox,
            outputs=[text_to_audio.tts_text_to_speech.text_input.textbox])
        
        text_to_audio.tts_text_to_speech.audio_output.change(
            fn=lambda x: x,
            inputs=text_to_audio.tts_text_to_speech.audio_output,
            outputs=text_to_video.video_creator.audio_uploaders[0],
        )
                                                                 

if __name__ == "__main__":
    with gr.Blocks() as interface:
        n2v = NewsToVideoPro()
    interface.launch()