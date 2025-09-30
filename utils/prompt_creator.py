# app.py
from llm_interface import LLMInterface
import gradio as gr

base_prompt = """Ты — эксперт по визуализации текста и созданию иллюстрационных промптов.
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

Итог должен быть списком промптов для генерации иллюстраций, которые можно подавать напрямую в графическую модель."""

styles = [
    "реализм",
    "книжная иллюстрация",
    "цифровая живопись",
    "акварель",
    "комикс",
    "минимализм",
    "стилизованная графика"
]

illustration_interface = LLMInterface(
    title="Генератор иллюстрационных промптов",
    heading="Разделение текста и генерация промптов для иллюстраций",
    prompt_label="📌 Системный промпт",
    prompt_default=base_prompt.format(style="реализм"),
    input_label="Исходный текст",
    input_placeholder="Вставьте сюда длинный текст...",
    generate_button_text="✨ Сгенерировать промпты",
    output_label="Список промптов для иллюстраций"
)

if __name__ == "__main__":
    models = illustration_interface.get_models()

    with gr.Blocks(title=illustration_interface.title) as demo:
        gr.Markdown(f"## 📝 {illustration_interface.heading}")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=models,
                value=models[0] if models else None,
                label="Выберите модель"
            )
            style_dropdown = gr.Dropdown(
                choices=styles,
                value="реализм",
                label="Выберите художественный стиль"
            )

        prompt_box = gr.Textbox(
            label=illustration_interface.prompt_label,
            value=illustration_interface.prompt_default,
            lines=7
        )
        input_box = gr.Textbox(
            label=illustration_interface.input_label,
            placeholder=illustration_interface.input_placeholder,
            lines=5
        )
        run_button = gr.Button(illustration_interface.generate_button_text)
        output_box = gr.Textbox(
            label=illustration_interface.output_label,
            lines=15
        )
        scene_dropdown = gr.Dropdown(choices=[], label="Выберите сцену")
        scene_prompt_box = gr.Textbox(label="Промпт выбранной сцены", lines=10, show_copy_button=True)
        segments_state = gr.State([])

        # Обновление промпта при смене стиля
        def update_prompt(style):
            return base_prompt.format(style=style)
        style_dropdown.change(fn=update_prompt, inputs=style_dropdown, outputs=prompt_box)

        # Генерация промптов и формирование сегментов
        def generate_prompts(model_name, prompt, input_text):
            full_output = illustration_interface.generate(model_name, prompt, input_text)
            segments = []
            titles = []
            current_segment = ""
            for line in full_output.split("\n"):
                if line.startswith("**Сегмент"):
                    if current_segment:
                        segments.append(current_segment.strip())
                    current_segment = line + "\n"
                    titles.append(line)
                else:
                    current_segment += line + "\n"
            if current_segment:
                segments.append(current_segment.strip())
            return full_output, titles, segments

        # При нажатии кнопки Генерировать
        def run_generation(model_name, prompt, input_text):
            full_output, titles, segments = generate_prompts(model_name, prompt, input_text)
            return full_output, gr.update(choices=titles, value=titles[0] if titles else None), segments


        run_button.click(
            fn=run_generation,
            inputs=[model_dropdown, prompt_box, input_box],
            outputs=[output_box, scene_dropdown, segments_state]
        )


        # Отображение промпта выбранной сцены
        def show_scene_prompt(selected_title, segments):
            if not selected_title or not segments:
                return ""
            
            # Находим индекс сегмента, независимо от пробела после "Сегмент"
            import re
            match = re.search(r"Сегмент\s*(\d+)", selected_title)
            if not match:
                return ""
            index = int(match.group(1)) - 1
            if index >= len(segments):
                return ""
            
            segment_text = segments[index]
            lines = segment_text.split("\n")
            
            # Находим строку с "Иллюстрационный промпт" и возвращаем всё, что после неё
            for i, line in enumerate(lines):
                if "Иллюстрационный промпт" in line:
                    return "\n".join(lines[i+1:]).strip()
            
            # Если строка не найдена, возвращаем пустую строку
            return ""
        
        def show_scene_prompt2(selected_title, segments):
            if not selected_title or not segments:
                return ""
            index = int(selected_title.split("Сегмент ")[1].split()[0]) - 1
            if index >= len(segments):
                return ""
            segment_text = segments[index]
            # Разбиваем на строки
            lines = segment_text.split("\n")
            # Найти строку с '- **Иллюстрационный промпт:**' и взять всё после неё
            try:
                start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("- **Иллюстрационный промпт:**"))
                cleaned_lines = lines[start_idx + 1:]
            except StopIteration:
                # Если строки нет, возвращаем весь сегмент
                cleaned_lines = lines
            return "\n".join(cleaned_lines).strip()


        scene_dropdown.change(
            fn=show_scene_prompt,
            inputs=[scene_dropdown, segments_state],
            outputs=scene_prompt_box
        )

    demo.launch()
