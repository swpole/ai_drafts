import gradio as gr
from omogre import Accentuator, Transcriptor

# Инициализация модели (выполняется один раз при запуске скрипта)
print("Загрузка модели omogre...")
accentuator = Accentuator(data_path='omogre_data')
print("Модель загружена успешно!")

def accent_text(input_text):
    """
    Функция для расстановки ударений во входном тексте
    """
    if not input_text.strip():
        return "Введите текст для расстановки ударений"
    
    try:
        # Обрабатываем весь текст как одно предложение
        sentences = [input_text]
        accentuated = accentuator(sentences)
        return accentuated[0]
    except Exception as e:
        return f"Произошла ошибка при обработке: {e}"

# Создание Gradio-интерфейса
with gr.Blocks(title="Расстановка ударений") as demo:
    gr.Markdown("# Сервис расстановки ударений в тексте")
    gr.Markdown("Введите текст в левое поле и нажмите кнопку для расстановки ударений")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Исходный текст",
                placeholder="Введите текст для расстановки ударений здесь...",
                lines=10,
                max_lines=100,
                show_copy_button=True
            )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Текст с ударениями", 
                interactive=False,
                lines=10,
                max_lines=100,
                show_copy_button=True
            )
    
    accent_button = gr.Button("Расставить ударения", variant="primary")
    
    # Обработчик нажатия кнопки
    accent_button.click(
        fn=accent_text,
        inputs=input_text,
        outputs=output_text
    )
    
    # Добавляем обработку нажатия Enter в поле ввода
    input_text.submit(
        fn=accent_text,
        inputs=input_text,
        outputs=output_text
    )

# Запуск интерфейса
if __name__ == "__main__":
    demo.launch()