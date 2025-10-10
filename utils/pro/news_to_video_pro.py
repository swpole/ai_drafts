import gradio as gr
from news_summarizer_pro import NewsSummarizerPro
from text_to_audio_pro import TextToAudioPro
from text_to_video_pro import TextToVideoPro

class NewsToVideoPro:

    def __init__(self):
        self.create_interface()

    def create_interface(self):

        news_summerizer_pro = NewsSummarizerPro()

        text_to_audio_pro = TextToAudioPro()

        text_to_video_pro = TextToVideoPro()  

        self.allowed_path=text_to_video_pro.illustrator.allowed_paths      

        news_summerizer_pro.llm_interface.output_box.textbox.change(
            fn = lambda x: [x,x,x,x],
            inputs=news_summerizer_pro.llm_interface.output_box.textbox,
            outputs=[text_to_audio_pro.tts_text_to_speech.text_input.textbox, 
                     text_to_audio_pro.tts_vibevoice_from_demo.text_input.textbox,
                     text_to_audio_pro.tts_vibevoice_from_workflow.text_input.textbox,
                     text_to_video_pro.illustrator.illustration_prompt_generator.llm_interface.input_box.textbox])
        
        text_to_audio_pro.tts_text_to_speech.audio_output.change(
            fn=lambda x: x,
            inputs=text_to_audio_pro.tts_text_to_speech.audio_output,
            outputs=text_to_video_pro.video_creator.audio_uploaders[0],
        )

        text_to_audio_pro.tts_vibevoice_from_demo.complete_audio_output.change(
            fn=lambda x: x,
            inputs=text_to_audio_pro.tts_vibevoice_from_demo.complete_audio_output,
            outputs=text_to_video_pro.video_creator.audio_uploaders[0],
        )

        text_to_audio_pro.tts_vibevoice_from_workflow.output_audio.change(
            fn=lambda x: x,
            inputs=text_to_audio_pro.tts_vibevoice_from_workflow.output_audio,
            outputs=text_to_video_pro.video_creator.audio_uploaders[0],
        )
                                                                 

if __name__ == "__main__":
    with gr.Blocks() as interface:
        n2v = NewsToVideoPro()
    interface.launch(allowed_paths=n2v.allowed_path)