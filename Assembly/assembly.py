import gradio as gr
from moviepy import ImageClip, concatenate_videoclips
import tempfile
import os

def insert_image(gallery_images, new_img, selected_index, before: bool):
    """
    Вставляет new_img либо перед (если before=True), либо после (if before=False)
    позиции selected_index в gallery_images.
    Если selected_index is None или вне диапазона, просто добавляет в конец.
    """
    imgs = list(gallery_images) if gallery_images else []
    if not new_img:
        return imgs
    # Если выбранный индекс валиден
    if selected_index is not None and 0 <= selected_index < len(imgs):
        if before:
            imgs.insert(selected_index, new_img)
        else:
            imgs.insert(selected_index + 1, new_img)
    else:
        # если индекс невалиден — добавим в конец
        imgs.append(new_img)
    return imgs

def delete_image(gallery_images, selected_index):
    """
    Удаляет картинку по выбранному индексу.
    Если индекс невалиден, возвращает исходную галерею без изменений.
    """
    imgs = list(gallery_images) if gallery_images else []
    
    # Проверяем валидность индекса
    if selected_index is not None and 0 <= selected_index < len(imgs):
        imgs.pop(selected_index)
    
    return imgs

def create_video_with_transitions(gallery_images, duration_per_image, transition_duration, output_filename="output_video.mp4"):
    """
    Создает видео из изображений с эффектами перехода
    """
    if not gallery_images:
        raise gr.Error("Галерея пуста! Добавьте изображения для создания видео.")
    
    if duration_per_image <= 0:
        raise gr.Error("Длительность изображения должна быть положительной.")
    
    if transition_duration < 0:
        raise gr.Error("Длительность перехода не может быть отрицательной.")
    
    # Проверяем, чтобы переход не был длиннее половины времени показа изображения
    if transition_duration * 2 >= duration_per_image:
        raise gr.Error("Длительность перехода слишком велика. Она должна быть меньше половины длительности изображения.")
    
    clips = []
    
    for i, img_info in enumerate(gallery_images):
        # Получаем путь к файлу из объекта изображения галереи
        if isinstance(img_info, tuple) and len(img_info) == 2:
            img_path = img_info[0]  # Градио хранит изображения как (путь, метка)
        else:
            img_path = img_info
        
        # Создаем клип с указанной длительностью
        clip = ImageClip(img_path, duration=duration_per_image).resized(width=640, height=480)
        
        clips.append(clip)
    
    # Объединяем клипы
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # Сохраняем видео
    final_clip.write_videofile(output_filename, fps=24, logger=None)
    
    return output_filename

def on_select(evt: gr.SelectData):
    return evt.index

with gr.Blocks() as demo:
    gr.Markdown("# 🎬 Создатель видео из галереи")
    
    with gr.Row():
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Галерея изображений", columns=6, interactive=True, height=400)
            selected = gr.State(value=None)
            
            with gr.Row():
                new_img_upload = gr.Image(sources="upload", type="filepath", label="Загрузите картинку", height=100)
                new_img_url = gr.Textbox(label="Или введите URL", placeholder="https://...")
            
            with gr.Row():
                btn_before = gr.Button("Добавить перед текущим")
                btn_after = gr.Button("Добавить после текущего")
                btn_delete = gr.Button("Удалить текущий", variant="stop")
        
        with gr.Column(scale=1):
            gr.Markdown("### Настройки видео")
            duration_per_image = gr.Slider(minimum=1, maximum=10, value=3, step=0.5, label="Длительность каждого изображения (секунды)")
            transition_duration = gr.Slider(minimum=0, maximum=2, value=0.5, step=0.1, label="Длительность перехода (секунды)")
            output_filename = gr.Textbox(value="output_video.mp4", label="Имя выходного файла")
            
            btn_create_video = gr.Button("🎥 Создать видео", variant="primary", size="lg")
            
            video_output = gr.Video(label="Результат")
            
            gr.Markdown("""
            **Инструкция:**
            1. Добавьте изображения в галерею
            2. Настройте длительность и переходы
            3. Нажмите "Создать видео"
            4. Скачайте готовое видео
            """)

    # Обработчики для управления галереей
    gallery.select(on_select, [], selected)

    def add_before_handler(gallery_images, upload, url, sel_idx):
        new_img = upload if upload is not None else (url if url else None)
        return insert_image(gallery_images, new_img, sel_idx, before=True)

    btn_before.click(
        add_before_handler,
        inputs=[gallery, new_img_upload, new_img_url, selected],
        outputs=gallery
    )

    def add_after_handler(gallery_images, upload, url, sel_idx):
        new_img = upload if upload is not None else (url if url else None)
        return insert_image(gallery_images, new_img, sel_idx, before=False)

    btn_after.click(
        add_after_handler,
        inputs=[gallery, new_img_upload, new_img_url, selected],
        outputs=gallery
    )

    def delete_handler(gallery_images, sel_idx):
        return delete_image(gallery_images, sel_idx)

    btn_delete.click(
        delete_handler,
        inputs=[gallery, selected],
        outputs=gallery
    )

    # Обработчик для создания видео
    def create_video_handler(gallery_images, duration, transition, filename):
        try:
            video_path = create_video_with_transitions(gallery_images, duration, transition, filename)
            return video_path
        except Exception as e:
            raise gr.Error(f"Ошибка при создании видео: {str(e)}")

    btn_create_video.click(
        create_video_handler,
        inputs=[gallery, duration_per_image, transition_duration, output_filename],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch()