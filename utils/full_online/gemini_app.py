import gradio as gr
from google import genai
from google.genai import types

class GeminiApp:
    def __init__(self):
        self.client = genai.Client()

    def generate(self, prompt, system_instruction, temperature, top_p, top_k, max_output_tokens, safety_threshold):
        # Настройка safety
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=getattr(types.HarmBlockThreshold, safety_threshold),
            ),
        ]

        # Конфигурация
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            #temperature=temperature,
            #top_p=top_p,
            #top_k=top_k,
            #max_output_tokens=max_output_tokens,
            #safety_settings=safety_settings,
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config=config,
            contents=prompt,
        )
        return response.text

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("## 🌟 Gemini Generator with Config")

            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Ввод запроса", placeholder="Введите текст...")

                    with gr.Accordion("Основные настройки", open=True):
                        system_instruction = gr.Textbox(label="System Instruction", value="")
                        temperature = gr.Slider(0.0, 1.0, step=0.1, value=0.7, label="Temperature")
                        top_p = gr.Slider(0.0, 1.0, step=0.05, value=0.9, label="Top P")
                        top_k = gr.Slider(1, 100, step=1, value=40, label="Top K")
                        max_output_tokens = gr.Slider(10, 2048, step=10, value=512, label="Max Output Tokens")

                        with gr.Accordion("Дополнительные настройки", open=False):
                            safety_threshold = gr.Dropdown(
                                choices=["BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH"],
                                value="BLOCK_LOW_AND_ABOVE",
                                label="Safety Threshold"
                            )

                    run_btn = gr.Button("Сгенерировать")

                with gr.Column():
                    output = gr.Textbox(label="Результат", lines=10)

            run_btn.click(
                self.generate,
                inputs=[prompt, system_instruction, temperature, top_p, top_k, max_output_tokens, safety_threshold],
                outputs=[output]
            )

        demo.launch()


if __name__ == "__main__":
    app = GeminiApp()
    app.launch()
