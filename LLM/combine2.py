# simple_pipeline.py
import gradio as gr
from SpeechHandler import SpeechHandler
from speech_to_text_app import SpeechToTextApp
from audio_text_input import AudioTextInput
from LLMHandler import LLMHandler

speech_whisper = SpeechHandler("tiny")
speech_google = SpeechToTextApp()
speech_combo = AudioTextInput()
llm = LLMHandler()
text_shared = "Hi from main!"
def track_tab_change(tab_index):
    
    return

def main():
    """
    Простой запуск обоих интерфейсов в разных вкладках с отслеживанием переключения
    """
    
    # Создаем общий блок для добавления функционала отслеживания
    with gr.Blocks(title="Speech-to-LLM Pipeline") as main_interface:
        gr.Markdown("# 🎙️ → 🤖 Speech-to-LLM Pipeline")
        
        # Создаем TabbedInterface внутри основного блока
        with gr.Tabs() as tabs:
            with gr.TabItem("🎙️ Распознавание речи (Whisper)", id=0) as tab1:
                speech_whisper_interface = speech_whisper.create_interface()
                #speech_whisper_interface.render()

            with gr.TabItem("🎙️ Распознавание речи (Google)", id=1) as tab2:
                speech_google_interface = speech_google.create_interface()
                #speech_google_interface.render()

            with gr.TabItem("🎙️ Распознавание речи (Google+Wisper)", id=2) as tab3:
                speech_combo_interface = speech_combo.create_interface()
                #speech_combo_interface.render()
            
            with gr.TabItem("🤖 AI-ассистент (LLMHandler)", id=3) as tab4:
                llm_interface = llm.create_interface()
                #llm_interface.render()
        
        # Отслеживаем изменение вкладок
        tabs.select(
            fn=track_tab_change,
            inputs=[],
            outputs=[]
        )
    
    print("Запуск комбинированного интерфейса...")
    print("Откройте http://localhost:7860")
    main_interface.launch(server_port=7860)

if __name__ == "__main__":
    main()