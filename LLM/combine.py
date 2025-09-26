# simple_pipeline.py
import gradio as gr
from SpeechHandler import SpeechHandler
from LLMHandler import LLMHandler

def main():
    """
    Простой запуск обоих интерфейсов в разных вкладках
    """
    speech = SpeechHandler("base")
    llm = LLMHandler()
    
    # Создаем интерфейс с вкладками
    interface = gr.TabbedInterface(
        [
            speech.create_interface(),
            llm.create_interface()
        ],
        [
            "🎙️ Распознавание речи (SpeechHandler)",
            "🤖 AI-ассистент (LLMHandler)" 
        ],
        title="Speech-to-LLM Pipeline"
    )
    
    print("Запуск комбинированного интерфейса...")
    print("Откройте http://localhost:7860")
    interface.launch(server_port=7860)

if __name__ == "__main__":
    main()