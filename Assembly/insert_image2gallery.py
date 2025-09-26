import gradio as gr

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

def on_select(evt: gr.SelectData):
    return evt.index

with gr.Blocks() as demo:
    gallery = gr.Gallery(label="Галерея", columns=6, interactive=True)
    selected = gr.State(value=None)

    new_img_upload = gr.Image(sources="upload", type="filepath", label="Загрузите картинку",height=100)
    new_img_url = gr.Textbox(label="Или введите URL", placeholder="https://...")

    btn_before = gr.Button("Добавить перед текущим")
    btn_after = gr.Button("Добавить после текущего")
    btn_delete = gr.Button("Удалить текущий", variant="stop")  # Красная кнопка для удаления

    gallery.select(on_select, [], selected)

    # Обработчик для кнопки “добавить перед”
    def add_before_handler(gallery_images, upload, url, sel_idx):
        new_img = upload if upload is not None else (url if url else None)
        return insert_image(gallery_images, new_img, sel_idx, before=True)

    btn_before.click(
        add_before_handler,
        inputs=[gallery, new_img_upload, new_img_url, selected],
        outputs=gallery
    )

    # Аналогичный обработчик для “добавить после”
    def add_after_handler(gallery_images, upload, url, sel_idx):
        new_img = upload if upload is not None else (url if url else None)
        return insert_image(gallery_images, new_img, sel_idx, before=False)

    btn_after.click(
        add_after_handler,
        inputs=[gallery, new_img_upload, new_img_url, selected],
        outputs=gallery
    )

    # Обработчик для кнопки удаления
    def delete_handler(gallery_images, sel_idx):
        return delete_image(gallery_images, sel_idx)

    btn_delete.click(
        delete_handler,
        inputs=[gallery, selected],
        outputs=gallery
    )

demo.launch()