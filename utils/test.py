import gradio as gr

class SimpleText:
    def __init__(self):
        # Перенесем создание текстового поля в create_interface
        pass
        
    def create_interface(self):
        # Теперь создаем и возвращаем текстовое поле
        self.text_box = gr.Textbox(
            label="Enter some text", 
            placeholder="Type here...", 
            lines=2
        )
        return self.text_box

class Test:
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft(), title="Test Interface") as interface:
            # Создаем первый текстовый блок
            gr.Markdown("## Text Input 1")
            t1_component = SimpleText()
            t1_textbox = t1_component.create_interface()
            
            # Создаем второй текстовый блок
            gr.Markdown("## Text Input 2")
            t2_component = SimpleText()
            t2_textbox = t2_component.create_interface()
            
            # Добавляем callback функцию
            def update_t2(value):
                return value  # Просто возвращаем то же значение для t2
            
            # Привязываем событие изменения t1 к обновлению t2
            t1_textbox.change(
                fn=update_t2,
                inputs=t1_textbox,
                outputs=t2_textbox
            )
            
        return interface
    
    def launch(self):
        interface = self.create_interface()
        interface.launch()

if __name__ == "__main__":
    Test().launch()