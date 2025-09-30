import gradio as gr
from nodes import *
from nodes import CLIPTextEncode
from nodes import CheckpointLoaderSimple
from nodes import EmptyLatentImage
from nodes import KSampler
from nodes import SaveImage
from nodes import VAEDecode
import folder_paths
import os
import sys
import torch

class ComfyUIGradioInterface:

    def __init__(self):
        self.checkpoint_files = self.get_checkpoint_files()
        # Автоматическое определение пути к папке output
        self.output_dir = self.get_output_directory()
    
    def get_output_directory(self):
        """Автоматически определяет путь к папке output"""
        try:
            # Способ 1: Используем API ComfyUI (предпочтительно)
            return folder_paths.get_output_directory()
        except:
            # Способ 2: Резервный вариант - относительный путь
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfyui_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'ComfyUI_portable', 'ComfyUI_windows_portable', 'ComfyUI'))
            output_dir = os.path.join(comfyui_root, 'output')
            
            # Создаем папку, если она не существует
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
    
    def get_checkpoint_files(self):
        """Получить список доступных контрольных точек"""
        checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(('.safetensors', '.ckpt', '.pth'))]
        return checkpoint_files
    
    def generate_image(self, ckpt_name, width, height, batch_size, positive_prompt, negative_prompt, seed, steps, cfg, sampler_name, scheduler, denoise, filename_prefix):
        """Основная функция генерации изображения"""
        
        try:
            # Загрузка модели
            checkpointloadersimple_4 = CheckpointLoaderSimple()
            checkpointloadersimple_4_output = checkpointloadersimple_4.load_checkpoint(ckpt_name=ckpt_name)
            
            # Создание латентного изображения
            emptylatentimage_5 = EmptyLatentImage()
            emptylatentimage_5_output = emptylatentimage_5.generate(
                width=width, 
                height=height, 
                batch_size=batch_size
            )
            
            # Кодирование позитивного промпта
            cliptextencode_6 = CLIPTextEncode()
            cliptextencode_6_output = cliptextencode_6.encode(
                text=positive_prompt, 
                clip=checkpointloadersimple_4_output[1]
            )
            
            # Кодирование негативного промпта
            cliptextencode_7 = CLIPTextEncode()
            cliptextencode_7_output = cliptextencode_7.encode(
                text=negative_prompt, 
                clip=checkpointloadersimple_4_output[1]
            )
            
            # Сэмплирование
            ksampler_3 = KSampler()
            ksampler_3_output = ksampler_3.sample(
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                model=checkpointloadersimple_4_output[0],
                positive=cliptextencode_6_output[0],
                negative=cliptextencode_7_output[0],
                latent_image=emptylatentimage_5_output[0]
            )
            
            # Декодирование VAE
            vaedecode_8 = VAEDecode()
            vaedecode_8_output = vaedecode_8.decode(
                samples=ksampler_3_output[0], 
                vae=checkpointloadersimple_4_output[2]
            )
            
            # Сохранение изображения
            saveimage_9 = SaveImage()
            saveimage_9_output = saveimage_9.save_images(
                filename_prefix=filename_prefix, 
                images=vaedecode_8_output[0].detach()
            )
            
            # Возвращаем путь к сохраненному изображению
            output_dir = folder_paths.get_output_directory()
            output_files = saveimage_9_output['ui']['images']
            if output_files:
                image_path = os.path.join(output_dir, output_files[0]['filename'])
                return image_path
            else:
                return None
                
        except Exception as e:
            print(f"Ошибка при генерации изображения: {e}")
            return None
    
    def create_interface(self):
        """Создание Gradio интерфейса"""
        
        with gr.Blocks(title="ComfyUI Image Generator") as interface:
            gr.Markdown("# ComfyUI Image Generator")
            gr.Markdown("Генерация изображений с использованием ComfyUI workflow")
            
            with gr.Row():
                with gr.Column():
                    # Основные параметры
                    ckpt_name = gr.Dropdown(
                        choices=self.checkpoint_files,
                        value=self.checkpoint_files[0] if self.checkpoint_files else "",
                        label="Checkpoint Model",
                        info="Выберите модель для генерации"
                    )
                    
                    width = gr.Number(value=512, label="Width", precision=0, minimum=64, maximum=2048)
                    height = gr.Number(value=512, label="Height", precision=0, minimum=64, maximum=2048)
                    batch_size = gr.Number(value=1, label="Batch Size", precision=0, minimum=1, maximum=8)
                    
                    positive_prompt = gr.Textbox(
                        value="The city in the night", 
                        label="Positive Prompt",
                        lines=3,
                        placeholder="Введите позитивный промпт..."
                    )
                    
                    negative_prompt = gr.Textbox(
                        value="text, watermark", 
                        label="Negative Prompt", 
                        lines=2,
                        placeholder="Введите негативный промпт..."
                    )
                    
                    filename_prefix = gr.Textbox(
                        value="ComfyUI",
                        label="Filename Prefix",
                        placeholder="Префикс для имени файла"
                    )
                
                with gr.Column():
                    # Аккордеон с дополнительными параметрами
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Number(
                            value=990700609792200, 
                            label="Seed", 
                            precision=0,
                            info="Сид для воспроизводимости"
                        )
                        
                        steps = gr.Number(
                            value=20, 
                            label="Steps", 
                            precision=0,
                            minimum=1,
                            maximum=100
                        )
                        
                        cfg = gr.Number(
                            value=8, 
                            label="CFG Scale", 
                            precision=1,
                            minimum=1,
                            maximum=20
                        )
                        
                        sampler_name = gr.Textbox(
                            value="euler",
                            label="Sampler",
                            info="Название сэмплера"
                        )
                        
                        scheduler = gr.Textbox(
                            value="normal",
                            label="Scheduler",
                            info="Планировщик"
                        )
                        
                        denoise = gr.Slider(
                            value=1.0,
                            label="Denoise",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            info="Уровень деннойзинга"
                        )
            
            # Кнопка генерации и вывод
            generate_btn = gr.Button("Generate Image", variant="primary")
            
            output_image = gr.Image(
                label="Generated Image",
                type="filepath",
                interactive=True,
                show_download_button=True
            )
            
            # Обработка событий
            generate_btn.click(
                fn=self.generate_image,
                inputs=[
                    ckpt_name, width, height, batch_size,
                    positive_prompt, negative_prompt,
                    seed, steps, cfg, sampler_name, scheduler, denoise,
                    filename_prefix
                ],
                outputs=output_image
            )
            
            # Информация
            gr.Markdown("---")
            gr.Markdown("### Информация")
            gr.Markdown("""
            - **Checkpoint**: Выбор модели для генерации
            - **Width/Height**: Размеры выходного изображения
            - **Batch Size**: Количество генерируемых изображений за раз
            - **Prompts**: Позитивный и негативный промпты
            - **Advanced Settings**: Дополнительные параметры генерации
            """)
        
        return interface, [self.output_dir]

def main():
    """Запуск Gradio интерфейса"""
    generator = ComfyUIGradioInterface()
    interface, allowed_paths = generator.create_interface()
    interface.launch(
            #share=True, 
            server_name="127.0.0.1",
            allowed_paths=allowed_paths  # Разрешаем автоматически найденный путь
        )

if __name__ == "__main__":
    main()