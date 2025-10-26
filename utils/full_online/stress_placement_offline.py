import gradio as gr
from stressrnn import StressRNN
#python -m pip install git+https://github.com/Desklop/StressRNN
from textbox_with_stt_final_online import TextboxWithSTTOnline

class StressPlacementOffline:

    def __init__(self, stress_sign="`"):
        self.stress_sign = stress_sign
        self.create_interface()

    def accent_text(self, input_text):
        """
        Функция для расстановки ударений во входном тексте
        """
        try:
            stress_rnn = StressRNN()
            stressed_text = stress_rnn.put_stress(input_text, stress_symbol="+", accuracy_threshold=0.75, replace_similar_symbols=True)
            stressed_text = stressed_text.replace("+", self.stress_sign)
            return stressed_text
        except Exception as e:
            return f"Произошла ошибка при обработке: {e}"

    # Создание Gradio-интерфейса
    def create_interface(self):
    
        gr.Markdown("### Сервис расстановки ударений в тексте")
        gr.Markdown("Введите текст в левое поле и нажмите кнопку для расстановки ударений")
        
        with gr.Row():
            with gr.Column():
                self.input_text = TextboxWithSTTOnline(
                    label="Исходный текст",
                    placeholder="Введите текст для расстановки ударений здесь...",
                    lines=4,
                    max_lines=10,
                )
            
            accent_button = gr.Button("Расставить ударения", variant="primary")

            with gr.Column():
                self.output_text = TextboxWithSTTOnline(
                    label="Текст с ударениями", 
                    lines=4,
                    max_lines=10,
                )
        
        
        
        # Обработчик нажатия кнопки
        accent_button.click(
            fn=self.accent_text,
            inputs=self.input_text.textbox,
            outputs=self.output_text.textbox
        )

# Запуск интерфейса
if __name__ == "__main__":
    with gr.Blocks(title="Расстановка ударений") as demo:
        StressPlacementOffline()
    demo.launch()