# simple_pipeline.py
import gradio as gr
from SpeechHandler import SpeechHandler
from LLMHandler import LLMHandler

speech = SpeechHandler("base")
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
            with gr.TabItem("🎙️ Распознавание речи (SpeechHandler)", id=0) as tab1:
                speech_interface = speech.create_interface()
                speech_interface.render()
            
            with gr.TabItem("🤖 AI-ассистент (LLMHandler)", id=1) as tab2:
                llm_interface = llm.create_interface()
                llm_interface.render()
        
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