# input - text
# output - audio

import gradio as gr
from gemini_audio_online import GeminiAudioOnline

#python -m pip install feedparser duckduckgo_search newspaper3k bs4 lxml[html_clean]

class TextToAudioOnline:
    def __init__(self):
        
        self.create_interface()


    # ---------- UI ----------
    def create_interface(self):

        self.tts_text_to_speech = GeminiAudioOnline()

        gr.HTML("""<div style='height: 2px; background: linear-gradient(90deg, transparent, #666, transparent); margin: 40px 0;'></div>""")

        

        return


if __name__ == "__main__":
    with gr.Blocks() as interface:
        news_summarizer = TextToAudioOnline()

    interface.launch()
