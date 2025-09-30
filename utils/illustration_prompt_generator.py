# app.py
from llm_interface import LLMInterface
import gradio as gr
import re

class IllustrationPromptGenerator:
    def __init__(self):
        self.base_prompts = {
            "Детализированный иллюстрационный": """Ты — эксперт по визуализации текста и созданию иллюстрационных промптов.
Твоя задача — превратить длинный текст в серию промптов для генерации изображений.

Правила работы:
1. Разбей текст на последовательные сегменты, каждый примерно по 4 секунды чтения (≈50–60 слов).
2. Для каждого сегмента:
   - Выведи сам текст сегмента.
   - Составь иллюстрационный промпт, включающий:
     • описание сцены, персонажей и их действий;
     • атмосферу и настроение (таинственно, радостно, тревожно и т.д.);
     • художественный стиль: {style};
     • детали окружения и фона;
     • цветовую гамму и композицию (если уместно).

Формат ответа:

**Сегмент 1 (текст)**
- Иллюстрационный промпт: …

**Сегмент 2 (текст)**
- Иллюстрационный промпт: …

…и так далее, пока не будет обработан весь текст.

Итог должен быть списком промптов для генерации иллюстраций, которые можно подавать напрямую в графическую модель.""",

            "Кинематографический": """Ты — режиссер и художник-раскадровщик. Преврати текст в кинематографические промпты для генерации изображений.

Правила работы:
1. Разбей текст на сцены (каждая ≈50-60 слов).
2. Для каждой сцены укажи:
   - Текст оригинала
   - Визуальный промпт с элементами:
     • Композиция кадра (крупный план, общий план и т.д.)
     • Освещение и атмосфера
     • Ракурс и угол съемки
     • Художественный стиль: {style}
     • Эмоциональная нагрузка сцены

Формат ответа:

**Сцена 1 (текст)**
- Кинематографический промпт: …

**Сцена 2 (текст)**
- Кинематографический промпт: …

Создай визуально насыщенные описания, готовые для генерации изображений.""",

            "Cinematic (English)": """You are a director and storyboard artist. Transform text into cinematic prompts for image generation.

Working rules:
1. Break the text into scenes (each ≈50-60 words).
2. For each scene specify:
   - Original text
   - Visual prompt with elements:
     • Frame composition (close-up, wide shot, etc.)
     • Lighting and atmosphere
     • Camera angle and perspective
     • Artistic style: {style}
     • Emotional impact of the scene

Response format:

**Scene 1 (text)**
- Cinematic prompt: …

**Scene 2 (text)**
- Cinematic prompt: …

Create visually rich descriptions ready for image generation.""",

            "Detailed Illustration (English)": """You are an expert in text visualization and creating illustration prompts.
Your task is to transform long text into a series of image generation prompts.

Working rules:
1. Divide the text into consecutive segments, each about 4 seconds of reading (≈50–60 words).
2. For each segment:
   - Output the segment text itself.
   - Create an illustration prompt including:
     • description of the scene, characters and their actions;
     • atmosphere and mood (mysterious, joyful, anxious, etc.);
     • artistic style: {style};
     • environment and background details;
     • color palette and composition (where appropriate).

Response format:

**Segment 1 (text)**
- Illustration prompt: …

**Segment 2 (text)**
- Illustration prompt: …

…and so on until the entire text is processed.

The result should be a list of prompts for illustration generation that can be fed directly into a graphics model."""
        }
        
        self.styles_ru = [
            "реализм",
            "книжная иллюстрация",
            "цифровая живопись",
            "акварель",
            "комикс",
            "минимализм",
            "стилизованная графика"
        ]
        
        self.styles_en = [
            "realism",
            "book illustration", 
            "digital painting",
            "watercolor",
            "comic",
            "minimalism",
            "stylized graphics"
        ]

        self.prompt_types = list(self.base_prompts.keys())
        
        # Определяем язык для каждого типа промпта
        self.prompt_languages = {
            "Детализированный иллюстрационный": "ru",
            "Кинематографический": "ru", 
            "Cinematic (English)": "en",
            "Detailed Illustration (English)": "en"
        }
        
        self.illustration_interface = LLMInterface(
            title="Генератор иллюстрационных промптов",
            heading="Разделение текста и генерацию промптов для иллюстраций",
            prompt_label="📌 Системный промпт",
            prompt_default=self.base_prompts["Детализированный иллюстрационный"].format(style="реализм"),
            input_label="Исходный текст",
            input_placeholder="Вставьте сюда длинный текст...",
            generate_button_text="✨ Сгенерировать промпты",
            output_label="Список промптов для иллюстраций"
        )

    def get_styles_for_prompt_type(self, prompt_type):
        """Получение списка стилей в зависимости от типа промпта"""
        language = self.prompt_languages.get(prompt_type, "ru")
        if language == "en":
            return self.styles_en
        else:
            return self.styles_ru

    def get_default_style(self, prompt_type):
        """Получение стиля по умолчанию в зависимости от типа промпта"""
        language = self.prompt_languages.get(prompt_type, "ru")
        if language == "en":
            return "realism"
        else:
            return "реализм"

    def update_prompt(self, prompt_type, style):
        """Обновление промпта при смене типа промпта и стиля"""
        base_prompt = self.base_prompts.get(prompt_type, self.base_prompts["Детализированный иллюстрационный"])
        return base_prompt.format(style=style)

    def update_interface(self, prompt_type):
        """Обновление интерфейса при смене типа промпта"""
        styles = self.get_styles_for_prompt_type(prompt_type)
        default_style = self.get_default_style(prompt_type)
        updated_prompt = self.update_prompt(prompt_type, default_style)
        
        return (
            gr.update(choices=styles, value=default_style),  # style_dropdown
            updated_prompt  # prompt_box
        )

    def generate_prompts(self, model_name, prompt, input_text):
        """Генерация промптов и формирование сегментов"""
        full_output = self.illustration_interface.generate(model_name, prompt, input_text)
        segments = []
        titles = []
        current_segment = ""
        
        # Определяем паттерн для сегментов в зависимости от языка промпта
        segment_patterns = {
            "ru": r"\*\*Сегмент|\*\*Сцена",
            "en": r"\*\*Segment|\*\*Scene"
        }
        
        # Определяем язык текущего промпта
        current_language = "ru"  # по умолчанию
        for prompt_type, language in self.prompt_languages.items():
            if prompt_type in prompt:
                current_language = language
                break
        
        segment_pattern = segment_patterns.get(current_language, r"\*\*Сегмент|\*\*Сцена|\*\*Segment|\*\*Scene")
        
        for line in full_output.split("\n"):
            if re.search(segment_pattern, line, re.IGNORECASE):
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = line + "\n"
                titles.append(line)
            else:
                current_segment += line + "\n"
                
        if current_segment:
            segments.append(current_segment.strip())
            
        return full_output, titles, segments

    def run_generation(self, model_name, prompt_type, style, input_text):
        """Обработчик нажатия кнопки Генерировать"""
        prompt = self.update_prompt(prompt_type, style)
        full_output, titles, segments = self.generate_prompts(model_name, prompt, input_text)
        return full_output, gr.update(choices=titles, value=titles[0] if titles else None), segments, prompt

    def show_scene_prompt(self, selected_title, segments):
        """Отображение промпта выбранной сцены"""
        if not selected_title or not segments:
            return ""
        
        # Находим индекс сегмента, независимо от языка
        match = re.search(r"(Сегмент|Сцена|Segment|Scene)\s*(\d+)", selected_title, re.IGNORECASE)
        if not match:
            return ""
        index = int(match.group(2)) - 1
        if 0 <= index < len(segments):
            segment_text = segments[index]
            lines = segment_text.split("\n")
            
            # Ищем строку с промптом (разные варианты на разных языках)
            prompt_keywords = [
                "Иллюстрационный промпт", "Кинематографический промпт", 
                "Illustration prompt", "Cinematic prompt", "промпт:"
            ]
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in prompt_keywords):
                    return "\n".join(lines[i:]).strip()
        
        return segment_text if 0 <= index < len(segments) else ""

    def create_interface(self):
        """Создание и запуск Gradio интерфейса"""
        models = self.illustration_interface.get_models()

        with gr.Blocks(title=self.illustration_interface.title) as interface:
            gr.Markdown(f"## 📝 {self.illustration_interface.heading}")

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=models[0] if models else None,
                    label="Модель"
                )
                

                
            with gr.Accordion(self.illustration_interface.prompt_label, open=False):

                with gr.Row():
                    prompt_type_dropdown = gr.Dropdown(
                        choices=self.prompt_types,
                        value=self.prompt_types[0],
                        label="Тип промпта"
                    )
                    style_dropdown = gr.Dropdown(
                        choices=self.styles_ru,
                        value="реализм",
                        label="Стиль"
                    )                
                prompt_box = gr.Textbox(
                    label=self.illustration_interface.prompt_label,
                    value=self.illustration_interface.prompt_default,
                    lines=5,
                    max_lines=5
                )

            input_box = gr.Textbox(
                label=self.illustration_interface.input_label,
                placeholder=self.illustration_interface.input_placeholder,
                lines=5,
                max_lines=5,
                interactive=True, show_copy_button=True
            )
            
            run_button = gr.Button(self.illustration_interface.generate_button_text)
            
            output_box = gr.Textbox(
                label=self.illustration_interface.output_label,
                lines=5,
                max_lines=5, 
                interactive=True, 
                show_copy_button=True
            )
            
            scene_dropdown = gr.Dropdown(choices=[], label="Выберите сцену")
            
            self.scene_prompt_box = gr.Textbox(
                label="Промпт выбранной сцены", 
                lines=5,
                max_lines=5,
                interactive=True,
                show_copy_button=True
            )
            
            segments_state = gr.State([])

            # Обновление стилей и промпта при смене типа промпта
            prompt_type_dropdown.change(
                fn=self.update_interface,
                inputs=[prompt_type_dropdown],
                outputs=[style_dropdown, prompt_box]
            )
            
            # Обновление промпта при смене стиля
            style_dropdown.change(
                fn=self.update_prompt, 
                inputs=[prompt_type_dropdown, style_dropdown], 
                outputs=prompt_box
            )

            # При нажатии кнопки Генерировать
            run_button.click(
                fn=self.run_generation,
                inputs=[model_dropdown, prompt_type_dropdown, style_dropdown, input_box],
                outputs=[output_box, scene_dropdown, segments_state, prompt_box]
            )

            # Отображение промпта выбранной сцены
            scene_dropdown.change(
                fn=self.show_scene_prompt,
                inputs=[scene_dropdown, segments_state],
                outputs=self.scene_prompt_box
            )

        return interface, self.scene_prompt_box

if __name__ == "__main__":
    generator = IllustrationPromptGenerator()
    interface, p = generator.create_interface()
    interface.launch()