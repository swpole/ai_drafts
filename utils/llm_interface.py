import gradio as gr
import ollama
import subprocess

class LLMInterface:
    def __init__(
        self,
        title: str = "Title",
        heading: str = "Heading",
        prompt_label: str = "Промпт для модели",
        prompt_default: str = "Prompt for the model",
        input_label: str = "Label input",
        input_placeholder: str = "Вставьте сюда текст ...",
        input_value: str = "Hi! How are you?",
        generate_button_text: str = "Label button",
        output_label: str = "Label output",
    ):
 
        self.title = title
        self.heading = heading
        self.prompt_label = prompt_label
        self.prompt_default = prompt_default
        self.input_label = input_label
        self.input_placeholder = input_placeholder
        self.input_value = input_value
        self.generate_button_text = generate_button_text
        self.output_label = output_label

    def get_models(self):
        """Возвращает список установленных моделей Ollama."""
        try:
            # Это запустит Ollama если не была запущена
            subprocess.run(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            models_info = ollama.list()
            return [m["model"] for m in models_info["models"]]
        except Exception as e:
            err = f"Ошибка при получении списка моделей:{e} Возможно ollama не запущена. Запустите 'ollama list' в терминале."
            return [err]

    def generate(self, model_name, prompt, input_text):
        """Запускает ollama с выбранной моделью."""
        if not input_text or not input_text.strip():
            return "⚠️ " + self.input_label + " не может быть пустым."
        try:
            full_prompt = f"{prompt}\n\n{self.input_label}:\n{input_text}"
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Ошибка генерации: {e}"

    def build_interface(self):
        models = self.get_models()
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"## 📝 {self.heading}")

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=models[0] if models else None,
                    label="Выберите модель"
                )

            prompt_box = gr.Textbox(
                label=self.prompt_label,
                value=self.prompt_default,
                lines=1
            )

            input_box = gr.Textbox(
                label=self.input_label,
                placeholder=self.input_placeholder,
                value=self.input_value,
                lines=5
            )

            run_button = gr.Button(self.generate_button_text)

            output_box = gr.Textbox(
                label=self.output_label,
                lines=5
            )

            run_button.click(
                fn=self.generate,
                inputs=[model_dropdown, prompt_box, input_box],
                outputs=output_box
            )

        return demo

if __name__ == "__main__":
    # Пример: использовать класс с дефолтными значениями
    summarizer = LLMInterface()
    demo = summarizer.build_interface()
    demo.launch()
