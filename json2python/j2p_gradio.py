import gradio as gr
import time
import os


import json2python

def handle_file(file):
    if file is None:
        return gr.update(visible=False), gr.update(visible=True)
    with open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
    return gr.update(value=content, visible=True), gr.update(visible=False)

def process_text(inputfile):
    input_json = inputfile.name
    orig_name = os.path.basename(inputfile.name)
    output_py = orig_name.rsplit('.', 1)[0] + '.py'
    
    # Создаем конвертер и обрабатываем файл
    converter = json2python.ComfyUIJsonToPythonConverter()
    
    try:
        # Загружаем JSON
        workflow_data = converter.load_workflow_json(input_json)

        # Генерируем Python код
        python_code = converter.generate_python_code(workflow_data)
        
        # Сохраняем результат
        converter.save_python_code(python_code, output_py)
        
        gr.Info(f"Файл {orig_name} успешно преобразован и сохранён как {output_py}")
        
    except Exception as e:
        gr.Info(f"Error during conversion: {str(e)}")
        raise
    return python_code

def save_text(text):
    filename = f"processed_{int(time.time())}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

with gr.Blocks() as demo:
    file_upload = gr.File(label="Загрузите json файл", visible=True)
    text_output = gr.Textbox(label="json файл", visible=False, lines=10, interactive=False)
    processed_output = gr.Textbox(label="python файл", visible=False, lines=10)

    file_upload.change(fn=handle_file, inputs=file_upload, outputs=[text_output, file_upload])

    def reset():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    reset_btn = gr.Button("Очистить")
    reset_btn.click(reset, inputs=[], outputs=[text_output, file_upload, processed_output])

    process_btn = gr.Button("Преобразовать json в python")
    process_btn.click(process_text, inputs=file_upload, outputs=processed_output)
    process_btn.click(lambda: gr.update(visible=True), inputs=[], outputs=processed_output)

demo.launch()
