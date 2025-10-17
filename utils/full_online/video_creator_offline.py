#input - images and audio
#output - video

import gradio as gr
from moviepy import * #python -m pip install moviepy
import tempfile
import os

class VideoCreatorOffline:
    def __init__(self, num_parallel_tracks=2):
        """
        Класс для создания видео из изображений с аудиодорожками
        
        Args:
            num_parallel_tracks (int): Количество параллельных аудиотреков
        """
        self.num_parallel_tracks = num_parallel_tracks
        self.audio_tracks = [[] for _ in range(num_parallel_tracks)]

        self.create_interface()

        pass
        
    def insert_image(self, gallery_images, new_img, selected_index, before: bool):
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
                selected_index+=1
                imgs.insert(selected_index, new_img)
        else:
            # если индекс невалиден — добавим в конец
            imgs.append(new_img)
        return imgs

    def delete_image(self, gallery_images, selected_index):
        """
        Удаляет картинку по выбранному индексу.
        Если индекс невалиден, возвращает исходную галерею без изменений.
        """
        imgs = list(gallery_images) if gallery_images else []
        
        # Проверяем валидность индекса
        if selected_index is not None and 0 <= selected_index < len(imgs):
            imgs.pop(selected_index)
        
        return imgs

    def insert_audio(self, track_index, audio_files, new_audio, selected_index, before: bool):
        """
        Вставляет new_audio в указанный трек либо перед (если before=True), 
        либо после (if before=False) позиции selected_index.
        """
        audio_list = list(audio_files) if audio_files else []
        if not new_audio:
            return audio_list
        # Если выбранный индекс валиден
        if selected_index is not None and 0 <= selected_index < len(audio_list):
            if before:
                audio_list.insert(selected_index, new_audio)
            else:
                audio_list.insert(selected_index + 1, new_audio)
        else:
            # если индекс невалиден — добавим в конец
            audio_list.append(new_audio)
        return audio_list

    def delete_audio(self, track_index, audio_files, selected_index):
        """
        Удаляет аудио из указанного трека по выбранному индексу.
        """
        audio_list = list(audio_files) if audio_files else []
        
        # Проверяем валидность индекса
        if selected_index is not None and 0 <= selected_index < len(audio_list):
            audio_list.pop(selected_index)
        
        return audio_list

    def create_video_with_transitions(self, gallery_images, duration_per_image, 
                                    transition_duration, audio_tracks, output_filename="output_video.mp4"):
        """
        Создает видео из изображений с плавными переходами и объединенной звуковой дорожкой.
        """
        if not gallery_images:
            raise gr.Error("Галерея пуста! Добавьте изображения для создания видео.")
        
        if duration_per_image <= 0:
            raise gr.Error("Длительность изображения должна быть положительной.")
        
        if transition_duration < 0:
            raise gr.Error("Длительность перехода не может быть отрицательной.")
        
        if transition_duration * 2 >= duration_per_image:
            raise gr.Error("Длительность перехода слишком велика. Она должна быть меньше половины длительности изображения.")
        
        clips = []
        
        for i, img_info in enumerate(gallery_images):
            # Получаем путь к файлу
            if isinstance(img_info, tuple) and len(img_info) == 2:
                img_path = img_info[0]
            else:
                img_path = img_info
            
            # Создаем клип
            clip = (
                ImageClip(img_path, duration=duration_per_image)
                .resized(width=640, height=480).with_effects([vfx.CrossFadeIn(1), vfx.CrossFadeOut(1)])
            )
            
            clips.append(clip)
        
        # Склеиваем с учетом наложения
        final_clip = concatenate_videoclips(clips, method="compose", padding=-transition_duration)
        
        # Добавляем звуковые дорожки, если они есть
        audio_clips_to_mix = []
        
        for track_index, audio_files in enumerate(audio_tracks):
            if audio_files and len(audio_files) > 0:
                try:
                    track_audio_clips = []
                    for audio_file in audio_files:
                        if audio_file:  # Проверяем, что файл не пустой
                            audio_clip = AudioFileClip(audio_file)
                            track_audio_clips.append(audio_clip)
                    
                    if track_audio_clips:
                        # Объединяем все аудиоклипы в треке последовательно
                        if len(track_audio_clips) == 1:
                            track_audio = track_audio_clips[0]
                        else:
                            track_audio = concatenate_audioclips(track_audio_clips)
                        
                        # Обрезаем или зацикливаем аудио под длину видео
                        if track_audio.duration < final_clip.duration:
                            # Зацикливаем аудио, если оно короче видео
                            track_audio = track_audio.loop(duration=final_clip.duration)
                        else:
                            # Обрезаем аудио, если оно длиннее видео
                            track_audio = track_audio.subclipped(0, final_clip.duration)
                        
                        audio_clips_to_mix.append(track_audio)
                        
                except Exception as e:
                    raise gr.Error(f"Ошибка при обработке аудиофайлов в треке {track_index + 1}: {str(e)}")
        
        # Микшируем все аудиодорожки вместе
        if audio_clips_to_mix:
            if len(audio_clips_to_mix) == 1:
                final_audio = audio_clips_to_mix[0]
            else:
                # Создаем композитный аудиоклип из всех дорожек
                final_audio = CompositeAudioClip(audio_clips_to_mix)
            
            final_clip = final_clip.with_audio(final_audio)
        
        final_clip.write_videofile(output_filename, fps=24, logger=None)
        
        return output_filename

    def create_interface(self):
        """Создает и возвращает Gradio интерфейс"""
        
        def on_select(evt: gr.SelectData):
            return evt.index

        def on_audio_select(evt: gr.SelectData, track_index):
            return evt.index, track_index
        gr.Markdown("### 🎬 Video creator")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    self.new_img_upload = gr.Image(sources="upload", type="filepath", label="Загрузите картинку", height=100)
                    new_img_url = gr.Textbox(label="Или введите URL", placeholder="https://...")
                
                with gr.Row():
                    btn_before = gr.Button("Добавить перед текущим")
                    btn_after = gr.Button("Добавить после текущего")
                    btn_delete = gr.Button("Удалить текущий", variant="stop")   

                gr.Markdown("### Изображения")
                gallery = gr.Gallery(label="Галерея изображений", columns=6, interactive=True)
                selected_image = gr.State(value=None)
                

                
                gr.Markdown("### Звуковые дорожки")
                
                # Создаем контейнер для параллельных треков
                audio_tracks_ui = []
                selected_audios = []
                self.audio_uploaders = []
                
                for i in range(self.num_parallel_tracks):
                    with gr.Group():
                        gr.Markdown(f"#### Аудиодорожка {i+1}")
                        
                        audio_gallery = gr.Gallery(
                            label=f"Трек {i+1} (параллельное воспроизведение)", 
                            columns=4, 
                            interactive=True,
                            object_fit="contain",
                            height=150
                        )
                        audio_tracks_ui.append(audio_gallery)
                        
                        selected_audio = gr.State(value=None)
                        selected_audios.append(selected_audio)
                        
                        with gr.Row():
                            audio_upload = gr.Audio(sources="upload", type="filepath", label=f"Загрузите аудио для трека {i+1}")
                            self.audio_uploaders.append(audio_upload)
                        
                        with gr.Row():
                            btn_audio_before = gr.Button(f"Добавить в трек {i+1} перед текущим")
                            btn_audio_after = gr.Button(f"Добавить в трек {i+1} после текущего")
                            btn_audio_delete = gr.Button(f"Удалить из трека {i+1}", variant="stop")
                        
                        # Обработчики для каждого трека
                        audio_gallery.select(
                            lambda evt, idx=i: (evt.index, idx), 
                            [], 
                            [selected_audios[i], gr.State(value=i)]
                        )
                        
                        # Обработчики кнопок для каждого трека
                        btn_audio_before.click(
                            lambda audio_files, upload, sel_idx, track_idx=i: self.insert_audio(track_idx, audio_files, upload, sel_idx, before=True),
                            inputs=[audio_tracks_ui[i], self.audio_uploaders[i], selected_audios[i]],
                            outputs=audio_tracks_ui[i]
                        )

                        btn_audio_after.click(
                            lambda audio_files, upload, sel_idx, track_idx=i: self.insert_audio(track_idx, audio_files, upload, sel_idx, before=False),
                            inputs=[audio_tracks_ui[i], self.audio_uploaders[i], selected_audios[i]],
                            outputs=audio_tracks_ui[i]
                        )

                        btn_audio_delete.click(
                            lambda audio_files, sel_idx, track_idx=i: self.delete_audio(track_idx, audio_files, sel_idx),
                            inputs=[audio_tracks_ui[i], selected_audios[i]],
                            outputs=audio_tracks_ui[i]
                        )
            
            with gr.Column(scale=1):
                gr.Markdown("### Настройки видео")
                duration_per_image = gr.Slider(minimum=1, maximum=10, value=3, step=0.5, label="Длительность каждого изображения (секунды)")
                transition_duration = gr.Slider(minimum=0, maximum=2, value=0.5, step=0.1, label="Длительность перехода (секунды)")
                
                output_filename = gr.Textbox(value="output_video.mp4", label="Имя выходного файла")
                
                btn_create_video = gr.Button("🎥 Создать видео", variant="primary", size="lg")
                
                video_output = gr.Video(label="Результат")
                
                gr.Markdown(f"""
                **Инструкция:**
                1. Добавьте изображения в галерею
                2. Добавьте аудиодорожки в {self.num_parallel_tracks} параллельных трека
                3. Настройте длительность и переходы
                4. Нажмите "Создать видео"
                5. Скачайте готовое видео
                
                **О параллельных треках:**
                - Аудио в каждом треке воспроизводится последовательно
                - Все треки воспроизводятся параллельно (одновременно)
                - Идеально для фоновой музыки + голосовых комментариев
                """)

        # Обработчики для управления галереей изображений
        gallery.select(on_select, [], selected_image)

        btn_before.click(
            lambda gallery_images, upload, url, sel_idx: self.insert_image(gallery_images, upload or url, sel_idx, before=True),
            inputs=[gallery, self.new_img_upload, new_img_url, selected_image],
            outputs=gallery
        )

        btn_after.click(
            lambda gallery_images, upload, url, sel_idx: self.insert_image(gallery_images, upload or url, sel_idx, before=False),
            inputs=[gallery, self.new_img_upload, new_img_url, selected_image],
            outputs=gallery
        )

        btn_delete.click(
            self.delete_image,
            inputs=[gallery, selected_image],
            outputs=gallery
        )

        # Обработчик для создания видео
        def create_video_handler(gallery_images, duration, transition, *audio_track_data):
            try:
                # Собираем все аудиотреки
                all_audio_tracks = []
                for i in range(self.num_parallel_tracks):
                    audio_files = audio_track_data[i]
                    audio_paths = []
                    if audio_files:
                        for audio_info in audio_files:
                            if isinstance(audio_info, tuple) and len(audio_info) == 2:
                                audio_paths.append(audio_info[0])
                            else:
                                audio_paths.append(audio_info)
                    all_audio_tracks.append(audio_paths)
                
                # Получаем имя файла из последнего аргумента
                filename = audio_track_data[-1] if audio_track_data else "output_video.mp4"
                
                video_path = self.create_video_with_transitions(gallery_images, duration, transition, all_audio_tracks, filename)
                return video_path
            except Exception as e:
                raise gr.Error(f"Ошибка при создании видео: {str(e)}")

        # Подготавливаем входные данные для обработчика создания видео
        input_components = [gallery, duration_per_image, transition_duration]
        for i in range(self.num_parallel_tracks):
            input_components.append(audio_tracks_ui[i])
        input_components.append(output_filename)

        btn_create_video.click(
            create_video_handler,
            inputs=input_components,
            outputs=video_output
        )

        return

# Использование класса
if __name__ == "__main__":

    with gr.Blocks() as demo:
        video_creator = VideoCreatorOffline(num_parallel_tracks=2)        
            
    demo.launch()