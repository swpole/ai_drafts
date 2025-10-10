# input - text
# output - audio

import gradio as gr
from facebook_tts_gradio_pro import TextToSpeechPro
from vibevoice_from_workflow_pro import VibeVoiceWorkflowPro
from vibevoice_from_gradio_demo_pro import VibeVoiceDemoPro

#python -m pip install feedparser duckduckgo_search newspaper3k bs4 lxml[html_clean]

class TextToAudioPro:
    def __init__(self):
        
        self.create_interface()


    # ---------- UI ----------
    def create_interface(self):

        self.tts_text_to_speech = TextToSpeechPro()

        gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")

        self.tts_vibevoice_from_workflow = VibeVoiceWorkflowPro()

        gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")

        self.tts_vibevoice_from_demo = VibeVoiceDemoPro()

        

        return


if __name__ == "__main__":
    with gr.Blocks() as interface:
        news_summarizer = TextToAudioPro()

    interface.launch(allowed_paths=[news_summarizer.tts_vibevoice_from_workflow.allowed_paths])
