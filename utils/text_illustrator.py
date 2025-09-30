# integrated_app.py
import gradio as gr
from illustration_prompt_generator import IllustrationPromptGenerator
from image_generator_simple_stt import ImageGeneratorSimpleSTT

class IntegratedApp:
    def __init__(self):
        """Инициализация интеграционного приложения"""
        self.prompt_generator = IllustrationPromptGenerator()
        self.image_generator = ImageGeneratorSimpleSTT()
    
    def launch_integrated(self):
        """Запуск интегрированного приложения"""
        print("Запуск генератора промптов...")
        prompt_demo = self.prompt_generator.demo()
        
        print("Запуск генератора изображений...")
        image_interface, allowed_paths = self.image_generator.create_interface()
        
        print("Оба интерфейса запущены и готовы к работе!")
        
        # Запускаем оба интерфейса
        # Они будут работать на разных портах
        prompt_demo.launch(server_name="127.0.0.1", server_port=7860)
        image_interface.launch(server_name="127.0.0.1", server_port=7861, allowed_paths=allowed_paths)

def main():
    """Основная функция запуска"""
    app = IntegratedApp()
    app.launch_integrated()

if __name__ == "__main__":
    main()