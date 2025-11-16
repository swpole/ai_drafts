# llm creates text
# inpput - text
# output - text

import gradio as gr
import ollama
import subprocess
import gc
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
        typical_prompts: dict = {},
        prompt_params: dict = {},
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
        
        # Словарь для отслеживания загруженных моделей
        self.loaded_models = {}

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

    def load_model(self, model_name):
        """Загружает модель, если она еще не загружена"""
        try:
            if model_name not in self.loaded_models:
                print(f"🔄 Загружаем модель: {model_name}")
                # Проверяем, существует ли модель
                models_info = ollama.list()
                available_models = [m["model"] for m in models_info["models"]]
                
                if model_name not in available_models:
                    return f"❌ Модель {model_name} не найдена. Доступные модели: {', '.join(available_models)}"
                
                # Загружаем модель
                ollama.chat(model=model_name, messages=[{"role": "user", "content": "ping"}])
                self.loaded_models[model_name] = True
                print(f"✅ Модель {model_name} загружена")
            return None  # Успешная загрузка
        except Exception as e:
            return f"❌ Ошибка загрузки модели {model_name}: {e}"

    def unload_model(self, model_name):
        """Выгружает модель из памяти"""
        try:
            print(f"🗑️ Выгружаем модель: {model_name}")
            # Используем Ollama API для выгрузки модели
            subprocess.run(["ollama", f"list"])
            subprocess.run(["ollama", f"stop", f"{model_name}"])
            
            del self.loaded_models[model_name]
            
            # Принудительный сбор мусора
            gc.collect()
            print(f"✅ Модель {model_name} выгружена")
        except Exception as e:
            print(f"⚠️ Ошибка при выгрузке модели {model_name}: {e}")

    def unload_all_models(self):
        """Выгружает все загруженные модели"""
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, check=True)
        output = result.stdout
        lines = output.strip().split('\n')
        running_models = []
        # Пропускаем заголовок и обрабатываем остальные строки
        for line in lines[1:]:  # пропускаем первую строку с заголовками
            if line.strip():  # проверяем, что строка не пустая
                # Разделяем по пробелам и берем первый элемент (имя модели)
                model_name = line.split()[0]  # берем первый столбец - имя модели
                running_models.append(model_name)

        for model_name in running_models:
            self.unload_model(model_name)

    def generate(self, model_name, prompt, input_text):
        temperature = self.temperature_slider.value
        top_p = self.top_p_slider.value
        
        if not input_text or not input_text.strip():
            return "⚠️ " + self.input_label + " не может быть пустым."
        
        # Загружаем модель перед генерацией
        load_error = self.load_model(model_name)
        if load_error:
            return load_error
        
        try:
            full_prompt = f"{prompt}\n\n{self.input_label}:\n{input_text}"
            response = ollama.chat(
                model=model_name,
                options={"temperature": temperature, "top_p": top_p},
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Автоматически выгружаем модель после использования (опционально)
            self.unload_model(model_name)
            
            return response["message"]["content"]
            
        except Exception as e:
            # В случае ошибки также пытаемся выгрузить модель
            self.unload_model(model_name)
            return f"Ошибка генерации: {e}"

    def on_model_change(self, model_name):
        """Обработчик изменения модели - выгружает предыдущую модель"""
        # Можно добавить логику для выгрузки предыдущей модели
        # если нужно поддерживать только одну модель в памяти
        pass

    def build_interface(self):
        models = self.get_models()

        # стартовые значения по индексам
        first_prompt_key = self.default_prompt_key
        first_param = self.default_param_value
        initial_prompt_text = self.prompt_default

        gr.Markdown(f"### 📝 {self.heading}")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=models,
                value=models[0] if models else None,
                label="Выберите модель"
            )

        # Добавляем кнопки для управления памятью
        with gr.Accordion("🧠 Управление памятью", open=False):
            with gr.Row():
                load_model_btn = gr.Button("🔄 Загрузить выбранную модель")
                unload_model_btn = gr.Button("🗑️ Выгрузить выбранную модель")
                unload_all_btn = gr.Button("🧹 Выгрузить все модели")
                memory_status_btn = gr.Button("📊 Статус памяти")
            
            memory_status = gr.Textbox(
                label="Статус памяти",
                interactive=False,
                lines=2
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
                self.temperature_slider = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.1, label="Температура"
                )
                self.top_p_slider = gr.Slider(
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
            lines=5,
            max_lines=5
        )

        run_button = gr.Button(self.generate_button_text)

        self.output_box = TextboxWithSTTPro(
            label=self.output_label,
            lines=5,
            max_lines=5
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
                else:
                    if "{style}" in template and param_choice:
                        return template.replace("{style}", param_choice)
                    else:
                        return template
            return self.prompt_default

        # Функции для управления памятью
        def load_selected_model(model_name):
            result = self.load_model(model_name)
            return result if result else f"✅ Модель {model_name} загружена"

        def unload_selected_model(model_name):
            self.unload_model(model_name)
            return f"✅ Модель {model_name} выгружена"

        def unload_all_models_wrapper():
            self.unload_all_models()
            return "✅ Все модели выгружены из памяти"

        def get_memory_status():
            loaded_count = len(self.loaded_models)
            status = f"📊 Загружено моделей: {loaded_count}\n"
            status += f"🧠 Модели в памяти: {', '.join(self.loaded_models.keys()) if self.loaded_models else 'нет'}"
            return status

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

        # Обработчики для управления памятью
        load_model_btn.click(
            fn=load_selected_model,
            inputs=model_dropdown,
            outputs=memory_status
        )
        
        unload_model_btn.click(
            fn=unload_selected_model,
            inputs=model_dropdown,
            outputs=memory_status
        )
        
        unload_all_btn.click(
            fn=unload_all_models_wrapper,
            outputs=memory_status
        )
        
        memory_status_btn.click(
            fn=get_memory_status,
            outputs=memory_status
        )

        run_button.click(
            fn=self.generate,
            inputs=[model_dropdown, prompt_box.textbox, self.input_box.textbox],
            outputs=self.output_box.textbox
        )

        # Обработчик изменения модели
        model_dropdown.change(
            fn=self.on_model_change,
            inputs=model_dropdown
        )


if __name__ == "__main__":
    # пример: берем 2-й промпт ("Учитель") и 2-й параметр ("физика")
    with gr.Blocks() as demo:
        summarizer = LLMInterfacePro(
            typical_prompts={
                "Писатель": "Ты талантливый писатель и рассказчик. Пиши увлекательно, живо и с деталями, чтобы захватить внимание читателя. Составь {param} из приведённого текста.",
            }, 
            prompt_params={
                "Писатель": ["краткое резюме", "рассказ", "эссе", "статья", "поэма"],
            },
            default_prompt_index=0, 
            default_param_index=0
        )
        
    demo.launch()