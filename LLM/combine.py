# simple_pipeline.py
import gradio as gr
from SpeechHandler import SpeechHandler
from speech_to_text_app import SpeechToTextApp
from audio_text_input import AudioTextInput
from LLMHandler import LLMHandler

def main():
    """
    Простой запуск обоих интерфейсов в разных вкладках
    """
    speech_whisper = SpeechHandler("tiny")
    speech_google = SpeechToTextApp()
    speech_combo = AudioTextInput()
    llm = LLMHandler()
    
    # Создаем интерфейс с вкладками
    interface = gr.TabbedInterface(
        [
            speech_whisper.create_interface(),
            speech_google.create_interface(),
            speech_combo.create_interface(),
            llm.create_interface()
        ],
        [
            "🎙️ Распознавание речи (whisper)",
            "🎙️ Распознавание речи (Google)",
            "🎙️ Распознавание речи (Google+Whisper)",
            "🤖 AI-ассистент (LLMHandler)" 
        ],
        title="Speech-to-LLM Pipeline"
    )
    
    print("Запуск комбинированного интерфейса...")
    print("Откройте http://localhost:7860")
    interface.launch(server_port=7860)

if __name__ == "__main__":
    main()