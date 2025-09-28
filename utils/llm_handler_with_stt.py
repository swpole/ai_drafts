import requests
import subprocess
import os
import gradio as gr

from textbox_with_stt_final import TextboxWithSTT

class LLMHandler:
    """
    Класс для работы с языковыми моделями через Ollama
    """
    
    def __init__(self):
        self.available_models = self.get_installed_ollama_models()
    
    def get_installed_ollama_models(self):
        """
        Автоматически получает список установленных моделей Ollama через API.
        Возвращает список названий моделей.
        """
        models_list = []
        
        try:
            # Попробуем получить модели через API Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models_list = [model["name"] for model in data.get("models", [])]
                print(f"Найдены модели через API: {models_list}")
                return models_list
        except requests.exceptions.RequestException as e:
            print(f"API Ollama недоступно: {e}")
            # Пробуем альтернативный способ через командную строку
            pass
        
        try:
            # Альтернативный способ: через команду ollama list
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Пропускаем заголовок
                    if line.strip():
                        model_name = line.split()[0]
                        models_list.append(model_name)
                print(f"Найдены модели через CLI: {models_list}")
                return models_list
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"Команда ollama list не сработала: {e}")
            pass
        
        # Если ничего не работает, возвращаем пустой список или дефолтные модели
        print("Не удалось автоматически определить модели. Использую дефолтный список.")
        return ["llama3.1", "gemma2", "mistral"]
    
    def check_ollama_running(self):
        """
        Проверяет, запущен ли сервер Ollama.
        """
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama_server(self):
        """
        Пытается запустить сервер Ollama.
        """
        try:
            print("Попытка запуска сервера Ollama...")
            # Запускаем в фоновом режиме (зависит от ОС)
            if os.name == 'nt':  # Windows
                subprocess.Popen(["ollama", "serve"], 
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:  # Linux/Mac
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            # Даем серверу время на запуск
            import time
            time.sleep(3)
            return self.check_ollama_running()
        except Exception as e:
            print(f"Ошибка запуска сервера Ollama: {e}")
            return False
    
    def query_ollama(self, model_name, prompt, system_prompt="", temperature=0.7, max_tokens=1000):
        """
        Отправляет запрос (prompt) в выбранную модель Ollama и возвращает ответ.
        
        Args:
            model_name: название модели
            prompt: текст запроса
            system_prompt: системный промпт (роль модели)
            temperature: креативность (0-1)
            max_tokens: максимальное количество токенов
        """
        if not prompt.strip():
            return "Запрос пуст. Введите текст."
        
        # Проверяем, запущен ли сервер Ollama
        if not self.check_ollama_running():
            if not self.start_ollama_server():
                return "Сервер Ollama не запущен. Запустите Ollama вручную."

        print(f"Отправка запроса модели {model_name}")
        try:
            # Формируем полный промпт с системной инструкцией
            full_prompt = prompt
            if system_prompt.strip():
                full_prompt = f"{system_prompt}\n\nЗапрос: {prompt}"
            
            # Используем API Ollama для более надежного взаимодействия
            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", 
                                   json=payload, 
                                   timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Ответ не получен")
            else:
                return f"Ошибка API: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "Таймаут запроса к модели. Попробуйте позже."
        except requests.exceptions.RequestException as e:
            return f"Ошибка соединения с Ollama: {str(e)}"
        except Exception as e:
            return f"Неожиданная ошибка: {str(e)}"
    
    def refresh_models(self):
        """
        Обновляет список доступных моделей.
        """
        self.available_models = self.get_installed_ollama_models()
        return gr.Dropdown(choices=self.available_models)
    
    def create_interface(self):
        """
        Создает Gradio интерфейс для работы с LLM
        """
        with gr.Blocks(title="LLM Interface") as interface:
            gr.Markdown("# 🤖 Интерфейс для работы с локальными LLM моделями")
            
            # Проверка статуса Ollama
            ollama_status = "🟢 Запущен" if self.check_ollama_running() else "🔴 Не запущен"
            gr.Markdown(f"**Статус Ollama:** {ollama_status}")

            with gr.Row():
                # Выбор модели Ollama с кнопкой обновления
                with gr.Column(scale=3):
                    model_dropdown = gr.Dropdown(
                        choices=self.available_models, 
                        value=self.available_models[0] if self.available_models else None,
                        label="Выберите модель Ollama",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    refresh_button = gr.Button("🔄 Обновить список моделей")

            # Системный промпт (роль модели)
            system_prompt_stt = TextboxWithSTT()
            system_prompt = system_prompt_stt.render(
                label="Системный промпт (роль модели)",
                placeholder="Например: Ты - опытный политолог. Отвечай профессионально и аргументированно...",
                lines=3,
                value="Ты - полезный AI ассистент. Отвечай точно и информативно.",
                show_copy_button=True
            )


            # Текстовое поле для ввода задания
            with gr.Row():
                user_prompt_stt = TextboxWithSTT()
                user_prompt = user_prompt_stt.render(
                    label="Задание для LLM",
                    placeholder="Введите ваш запрос здесь...",
                    lines=5,
                    show_copy_button=True
                )


            # Аккордеон для дополнительных параметров
            with gr.Accordion("Дополнительные параметры модели", open=False):
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1,
                        label="Температура (креативность)",
                        info="Чем выше значение, тем более креативными будут ответы"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=100, 
                        maximum=4000, 
                        value=1000, 
                        step=100,
                        label="Максимальное количество токенов",
                        info="Ограничивает длину ответа модели"
                    )

            # Кнопка отправки запроса
            submit_button = gr.Button("🚀 Отправить запрос", variant="primary")

            # Блок для вывода ответа модели
            response_output_stt = TextboxWithSTT()
            response_output = response_output_stt.render(
                label="Ответ LLM", 
                lines=10, 
                interactive=True,
                show_copy_button=True
            )

            # Обработчики событий
            # Отправка запроса к LLM
            submit_button.click(
                fn=self.query_ollama, 
                inputs=[
                    model_dropdown, 
                    user_prompt, 
                    system_prompt,
                    temperature_slider,
                    max_tokens_slider
                ], 
                outputs=response_output
            )

            # Обновление списка моделей
            refresh_button.click(
                fn=self.refresh_models,
                outputs=model_dropdown
            )

            # Примеры запросов
            with gr.Accordion("Примеры запросов", open=False):
                gr.Markdown("""
                **Примеры системных промптов:**
                - Ты - опытный программист. Помоги написать код и объясни решения.
                - Ты - историк. Расскажи факты точно и объективно.
                - Ты - креативный писатель. Придумай интересную историю.
                
                **Примеры заданий:**
                - Напиши Python функцию для сортировки списка
                - Объясни квантовую физику простыми словами
                - Придумай бизнес-план для стартапа
                """)

        return interface

    def launch_interface(self, server_name="0.0.0.0", share=False, debug=True):
        """
        Запускает Gradio интерфейс
        """
        interface = self.create_interface()
        
        print("Запуск интерфейса LLMHandler...")
        print("Откройте http://localhost:7860 в вашем браузере")
        
        try:
            interface.launch(
                server_name=server_name,
                share=share,
                debug=debug
            )
        except Exception as e:
            print(f"Ошибка запуска интерфейса: {e}")
            print("Проверьте, что порт 7860 свободен или укажите другой порт")

if __name__ == "__main__":
    # Создаем экземпляр обработчика LLM
    llm_handler = LLMHandler()
    
    print("=== LLMHandler Interface ===")
    print(f"Доступные модели: {llm_handler.available_models}")
    print(f"Сервер Ollama запущен: {llm_handler.check_ollama_running()}")
    
    # Проверяем доступность Ollama при запуске
    if not llm_handler.check_ollama_running():
        print("Сервер Ollama не запущен. Пытаюсь запустить...")
        if llm_handler.start_ollama_server():
            print("Сервер Ollama успешно запущен")
        else:
            print("Не удалось запустить сервер Ollama. Убедитесь, что Ollama установлен.")
    
    # Запускаем интерфейс
    llm_handler.launch_interface()