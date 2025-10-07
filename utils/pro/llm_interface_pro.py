import gradio as gr
import ollama
import subprocess
from textbox_with_stt_final_pro import TextboxWithSTTPro

class LLMInterfacePro:
    def __init__(
        self,
        title: str = "Title",
        heading: str = "Heading",
        prompt_label: str = "Промпт для модели",
        prompt_default: str = None,
        input_label: str = "Label input",
        input_placeholder: str = "Вставьте сюда текст ...",
        input_value: str = "Hi! How are you?",
        generate_button_text: str = "Label button",
        output_label: str = "Label output",
        typical_prompts: dict = None,
        prompt_params: dict = None,
        default_prompt_index: int = 0,
        default_param_index: int = 0,
    ):
        self.title = title
        self.heading = heading
        self.prompt_label = prompt_label
        self.input_label = input_label
        self.input_placeholder = input_placeholder
        self.input_value = input_value
        self.generate_button_text = generate_button_text
        self.output_label = output_label

        # --- дефолтные промпты и параметры ---
        self.typical_prompts = {
            "Эксперт-ассистент": "Ты дружелюбный эксперт и помощник в области {param}. Отвечай чётко, структурировано и добавляй примеры. Объясняй шаг за шагом, если задача сложная.",
            "Учитель": "Ты опытный учитель по предмету {param}. Объясняй материал простыми словами, приводя примеры из реальной жизни.",
            "Консультант": "Ты профессиональный консультант в сфере {param}. Сначала дай краткий ответ по делу, затем подробно объясни логику и варианты.",
            "Исследователь": "Ты аналитик-исследователь в области {param}. Делай разбор задачи, приводи аргументы, сравнивай разные подходы и делай вывод.",
            "Переводчик": "Переведи текст на {param}, сохрани стиль и контекст. Если есть неоднозначности, поясни их."
        }

        self.prompt_params = {
            "Эксперт-ассистент": ["ИИ", "программирование", "медицина", "экономика"],
            "Учитель": ["математика", "физика", "история", "литература"],
            "Консультант": ["бизнес", "маркетинг", "финансы", "стартапы"],
            "Исследователь": ["технологии", "наука", "политика", "общество"],
            "Переводчик": ["английский", "немецкий", "японский", "китайский"]
        }

        self.typical_prompts = {**typical_prompts, **self.typical_prompts}
        self.prompt_params = {**prompt_params, **self.prompt_params}

        # --- определяем дефолтный промпт и параметр по индексам ---
        keys = list(self.typical_prompts.keys())
        if default_prompt_index < 0 or default_prompt_index >= len(keys):
            default_prompt_index = 0
        self.default_prompt_key = keys[default_prompt_index]

        params = self.prompt_params.get(self.default_prompt_key, [""])
        if default_param_index < 0 or default_param_index >= len(params):
            default_param_index = 0
        self.default_param_value = params[default_param_index]

        # --- формируем дефолтный текст промпта ---
        if prompt_default:
            self.prompt_default = prompt_default
        else:
            self.prompt_default = self.typical_prompts[self.default_prompt_key].replace("{param}", self.default_param_value)

        self.build_interface()

    def get_models(self):
        try:
            subprocess.run(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            models_info = ollama.list()
            return [m["model"] for m in models_info["models"]]
        except Exception as e:
            err = f"Ошибка при получении списка моделей: {e} Возможно ollama не запущена."
            return [err]

    def generate(self, model_name, prompt, input_text, temperature, top_p):
        if not input_text or not input_text.strip():
            return "⚠️ " + self.input_label + " не может быть пустым."
        try:
            full_prompt = f"{prompt}\n\n{self.input_label}:\n{input_text}"
            response = ollama.chat(
                model=model_name,
                options={"temperature": temperature, "top_p": top_p},
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Ошибка генерации: {e}"

    def build_interface(self):
        models = self.get_models()

        # стартовые значения по индексам
        first_prompt_key = self.default_prompt_key
        first_param = self.default_param_value
        initial_prompt_text = self.prompt_default

        gr.Markdown(f"## 📝 {self.heading}")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=models,
                value=models[0] if models else None,
                label="Выберите модель"
            )

        with gr.Accordion("⚙️ Настройки промптов и модели", open=False):
            with gr.Row():
                typical_prompt_dd = gr.Dropdown(
                    choices=list(self.typical_prompts.keys()),
                    value=first_prompt_key,
                    label="Типичный промпт"
                )
                param_dd = gr.Dropdown(
                    choices=self.prompt_params.get(first_prompt_key, []),
                    value=first_param,
                    label="Параметр"
                )

            with gr.Row():
                temperature_slider = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.1, label="Температура"
                )
                top_p_slider = gr.Slider(
                    0.0, 1.0, value=0.9, step=0.1, label="Top-p"
                )

        prompt_box = TextboxWithSTTPro(
            label=self.prompt_label,
            value=initial_prompt_text,
            lines=2
        )



        self.input_box = TextboxWithSTTPro(
            label=self.input_label,
            placeholder=self.input_placeholder,
            value=self.input_value,
            lines=5
        )

        run_button = gr.Button(self.generate_button_text)

        output_box = TextboxWithSTTPro(
            label=self.output_label,
            lines=5
        )

        # функции для синхронизации
        def update_params(prompt_choice):
            return gr.update(
                choices=self.prompt_params.get(prompt_choice, []),
                value=self.prompt_params.get(prompt_choice, [""])[0]
            )

        def apply_prompt(prompt_choice, param_choice):
            if prompt_choice in self.typical_prompts:
                template = self.typical_prompts[prompt_choice]
                if "{param}" in template and param_choice:
                    return template.replace("{param}", param_choice)
                return template
            return self.prompt_default

        # обновление параметров и промптов
        typical_prompt_dd.change(
            fn=update_params,
            inputs=typical_prompt_dd,
            outputs=param_dd
        )
        typical_prompt_dd.change(
            fn=apply_prompt,
            inputs=[typical_prompt_dd, param_dd],
            outputs=prompt_box.textbox
        )
        param_dd.change(
            fn=apply_prompt,
            inputs=[typical_prompt_dd, param_dd],
            outputs=prompt_box.textbox
        )

        run_button.click(
            fn=self.generate,
            inputs=[model_dropdown, prompt_box.textbox, self.input_box.textbox, temperature_slider, top_p_slider],
            outputs=output_box.textbox
        )

        return 


if __name__ == "__main__":
    # пример: берем 2-й промпт ("Учитель") и 2-й параметр ("физика")
    with gr.Blocks() as demo:
        summarizer = LLMInterfacePro(typical_prompts={
            "Писатель": "Ты талантливый писатель и рассказчик. Пиши увлекательно, живо и с деталями, чтобы захватить внимание читателя. Составь {param} из приведённого текста.",
        }, 
        prompt_params={
            "Писатель": ["краткое резюме", "рассказ", "эссе", "статья", "поэма"],
        },
        default_prompt_index=0, default_param_index=0)
        
    demo.launch()
