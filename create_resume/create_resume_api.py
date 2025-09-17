import os
import sys

# добавляем родительскую папку в путь поиска модулей
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    utils_dir = os.path.join(parent_dir, 'utils')
    sys.path.insert(0, utils_dir)

from LLMInterface import LLMInterface

def main():
    # Создаем интерфейс так, чтобы он как в исходном — резюме статьи
    summarizer = LLMInterface(
        title="Summarizer with Ollama",
        heading="Резюме статьи",
        prompt_label="Промпт для модели",
        prompt_default="Составь краткое резюме этого текста.",
        input_label="Текст статьи",
        input_placeholder="Вставьте сюда текст статьи...",
        generate_button_text="Сгенерировать резюме",
        output_label="Сгенерированное резюме"
    )

    demo = summarizer.build_interface()
    demo.launch()

if __name__ == "__main__":
    main()