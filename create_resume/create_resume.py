import gradio as gr
import ollama


def get_models():
    """Возвращает список установленных моделей Ollama."""
    try:
        models_info = ollama.list()
        return [m["model"] for m in models_info["models"]]
    except Exception as e:
        print("Ошибка при получении списка моделей:", e)
        return ["llama2"]  # запасной вариант


def generate_summary(model_name, prompt, article_text):
    """Запускает ollama с выбранной моделью и формирует резюме."""
    if not article_text.strip():
        return "⚠️ Текст статьи."
    try:
        full_prompt = f"{prompt}\n\nТекст статьи:\n{article_text}"
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": full_prompt}
        ])
        return response["message"]["content"]
    except Exception as e:
        return f"Ошибка генерации: {e}"


def build_interface():
    models = get_models()
    with gr.Blocks(title="Summarizer with Ollama") as demo:
        gr.Markdown("## 📝 Резюме статьи")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=models,
                value=models[0] if models else None,
                label="Выберите модель"
            )

        prompt_box = gr.Textbox(
            label="Промпт для модели",
            value="Составь краткое резюме этого текста.",
            lines=1
        )

        article_box = gr.Textbox(
            label="Текст статьи",
            placeholder="Вставьте сюда текст статьи...",
            lines=5
        )

        run_button = gr.Button("Сгенерировать резюме")

        output_box = gr.Textbox(
            label="Сгенерированное резюме",
            lines=5
        )

        run_button.click(
            fn=generate_summary,
            inputs=[model_dropdown, prompt_box, article_box],
            outputs=output_box
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
