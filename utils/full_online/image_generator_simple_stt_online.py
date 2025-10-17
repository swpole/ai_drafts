# input - prompt
# output - image

import gradio as gr
import folder_paths
import os
from textbox_with_stt_final_online import TextboxWithSTTOnline

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

class ImageGeneratorSimpleSTTOnline:
    def __init__(self):
        self.create_interface()
        return

    
    def generate_image(self, positive_prompt):
        """Основная функция генерации изображения"""
        
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[positive_prompt],
            )

            image_path = "generated_image.png"
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(image_path)
        except Exception as e:
            print(f"Error generating image: {e}")
            return "Error"
            
        return image_path
    
    def create_interface(self):
        """Создание Gradio интерфейса"""
        

        gr.Markdown("### Image Generator (ComfyUI simple with STT)")
        gr.Markdown("Генерация изображений с использованием ComfyUI workflow")
        
        with gr.Row():
            with gr.Column():
                self.positive_prompt = TextboxWithSTTOnline(
                    label="Positive Prompt", 
                    lines=3, 
                    placeholder="Введите позитивный промпт...",
                    value="The city in the night"
                )

        # Кнопка генерации и вывод
        generate_btn = gr.Button("Generate Image", variant="primary")
        
        self.output_image = gr.Image(
            label="Generated Image",
            type="filepath",
            interactive=True,
            show_download_button=True
        )
        
        # Обработка событий
        generate_btn.click(
            fn=self.generate_image,
            inputs=[
                self.positive_prompt.textbox
            ],
            outputs=self.output_image
        )
        
        # Информация
        with gr.Accordion("More Information", open=False):
            gr.Markdown("""
            - **Checkpoint**: Выбор модели для генерации
            - **Width/Height**: Размеры выходного изображения
            - **Batch Size**: Количество генерируемых изображений за раз
            - **Prompts**: Позитивный и негативный промпты
            - **Advanced Settings**: Дополнительные параметры генерации
            """)
        
        return


if __name__ == "__main__":
    """Запуск Gradio интерфейса"""
    with gr.Blocks(title="ComfyUI Image Generator (simple with STT)") as interface:
        generator = ImageGeneratorSimpleSTTOnline()
        interface.launch(
                #share=True, 
                server_name="127.0.0.1",
            )