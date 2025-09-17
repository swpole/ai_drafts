import gradio as gr
import os, sys, json
from VibeVoice_example_api import run_vibevoice_tts

# ======== Функции для сохранения/загрузки настроек ========
# Папка, где находится скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Полный путь к файлу настроек
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_current_settings(text, speaker_1, speaker_2, speaker_3, speaker_4,
                         model_name, quantize_llm_4bit, attention_mode,
                         cfg_scale, inference_steps, seed, do_sample,
                         temperature, top_p, top_k, force_offload):
    return {
        "text": text,
        "speaker_1": speaker_1.name if speaker_1 else None,
        "speaker_2": speaker_2.name if speaker_2 else None,
        "speaker_3": speaker_3.name if speaker_3 else None,
        "speaker_4": speaker_4.name if speaker_4 else None,
        "model_name": model_name,
        "quantize_llm_4bit": quantize_llm_4bit,
        "attention_mode": attention_mode,
        "cfg_scale": cfg_scale,
        "inference_steps": inference_steps,
        "seed": seed,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "force_offload": force_offload
    }

# ======== Загружаем настройки ========
settings = load_settings()

default_text = settings.get("text", """Speaker 1: Не могу поверить, что ты снова это сделал. Я ждал два часа. Два часа! Ни одного звонка, ни одного сообщения. Ты хоть представляешь, как это было неловко — просто сидеть там в одиночестве?
Speaker 2: Слушай, я знаю, извини, ладно? Работа была настоящим кошмаром. Мой начальник в последнюю минуту поставил мне критически важный дедлайн. У меня даже секунды не было, чтобы вздохнуть, не говоря уже о том, чтобы проверить телефон.
Speaker 1: Кошмар? Это же оправдание ты использовал в прошлый раз. Я начинаю думать, что тебе просто всё равно. Легче сказать «работа была сумасшедшей», чем признать, что я больше не в приоритете для тебя.""")

default_speaker_1 = settings.get("speaker_1", r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac")
default_speaker_2 = settings.get("speaker_2", r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac")
default_speaker_3 = settings.get("speaker_3", r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac")
default_speaker_4 = settings.get("speaker_4", r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac")
default_model_name = settings.get("model_name", "VibeVoice-Large")
default_quantize_llm_4bit = settings.get("quantize_llm_4bit", False)
default_attention_mode = settings.get("attention_mode", "eager")
default_cfg_scale = settings.get("cfg_scale", 1.3)
default_inference_steps = settings.get("inference_steps", 10)
default_seed = settings.get("seed", 1)
default_do_sample = settings.get("do_sample", True)
default_temperature = settings.get("temperature", 0.95)
default_top_p = settings.get("top_p", 0.95)
default_top_k = settings.get("top_k", 0)
default_force_offload = settings.get("force_offload", False)

def load_settings_btn():
    settings = load_settings()  # читаем JSON
    return (
        settings.get("text", ""),
        settings.get("speaker_1", None),
        settings.get("speaker_2", None),
        settings.get("speaker_3", None),
        settings.get("speaker_4", None),
        settings.get("model_name", "VibeVoice-Large"),
        settings.get("quantize_llm_4bit", False),
        settings.get("attention_mode", "eager"),
        settings.get("cfg_scale", 1.3),
        settings.get("inference_steps", 10),
        settings.get("seed", 1),
        settings.get("do_sample", True),
        settings.get("temperature", 0.95),
        settings.get("top_p", 0.95),
        settings.get("top_k", 0),
        settings.get("force_offload", False)
    )

# ======== Основная функция для генерации речи и сохранения настроек ========
def tts_with_save(
    text, speaker_1, speaker_2, speaker_3, speaker_4,
    model_name, quantize_llm_4bit, attention_mode,
    cfg_scale, inference_steps, seed, do_sample,
    temperature, top_p, top_k, force_offload
):
    # Генерация речи
    result = run_vibevoice_tts(
        text=text,
        speaker_1_path=speaker_1.name if speaker_1 else None,
        speaker_2_path=speaker_2.name if speaker_2 else None,
        speaker_3_path=speaker_3.name if speaker_3 else None,
        speaker_4_path=speaker_4.name if speaker_4 else None,
        model_name=model_name,
        quantize_llm_4bit=quantize_llm_4bit,
        attention_mode=attention_mode,
        cfg_scale=cfg_scale,
        inference_steps=inference_steps,
        seed=seed,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        force_offload=force_offload
    )

    # Сохраняем текущие настройки
    settings = get_current_settings(
        text, speaker_1, speaker_2, speaker_3, speaker_4,
        model_name, quantize_llm_4bit, attention_mode,
        cfg_scale, inference_steps, seed, do_sample,
        temperature, top_p, top_k, force_offload
    )
    save_settings(settings)

    return result

# ======== Gradio интерфейс ========
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ VibeVoice TTS Demo")

    load_btn = gr.Button("Загрузить настройки")

    with gr.Row():
        # Голос 1
        with gr.Column(scale=0.25):
            speaker_1 = gr.File(label="Голос 1", type="filepath", value=default_speaker_1, elem_id="voice1")
            preview_1 = gr.Audio(label="▶", type="filepath", value=default_speaker_1, elem_id="preview1")
            speaker_1.change(lambda f: f.name if f else None, inputs=speaker_1, outputs=preview_1)

        # Голос 2
        with gr.Column(scale=0.25):
            speaker_2 = gr.File(label="Голос 2", type="filepath", value=default_speaker_2, elem_id="voice2")
            preview_2 = gr.Audio(label="▶", type="filepath", value=default_speaker_2, elem_id="preview2")
            speaker_2.change(lambda f: f.name if f else None, inputs=speaker_2, outputs=preview_2)

        # Голос 3
        with gr.Column(scale=0.25):
            speaker_3 = gr.File(label="Голос 3", type="filepath", value=default_speaker_3, elem_id="voice3")
            preview_3 = gr.Audio(label="▶", type="filepath", value=default_speaker_3, elem_id="preview3")
            speaker_3.change(lambda f: f.name if f else None, inputs=speaker_3, outputs=preview_3)

        # Голос 4
        with gr.Column(scale=0.25):
            speaker_4 = gr.File(label="Голос 4", type="filepath", value=default_speaker_4, elem_id="voice4")
            preview_4 = gr.Audio(label="▶", type="filepath", value=default_speaker_4, elem_id="preview4")
            speaker_4.change(lambda f: f.name if f else None, inputs=speaker_4, outputs=preview_4)

    # Текст для TTS
    text = gr.Textbox(label="Введите текст", lines=6, value=default_text)

    # Дополнительные параметры
    with gr.Accordion("Дополнительные параметры", open=False):
        model_name = gr.Textbox(value=default_model_name, label="Модель")
        quantize_llm_4bit = gr.Checkbox(value=default_quantize_llm_4bit, label="Quantize LLM 4bit")
        attention_mode = gr.Dropdown(choices=["eager", "flash"], value=default_attention_mode, label="Attention mode")
        cfg_scale = gr.Slider(minimum=0.1, maximum=3.0, value=default_cfg_scale, step=0.1, label="CFG Scale")
        inference_steps = gr.Slider(minimum=1, maximum=50, value=default_inference_steps, step=1, label="Inference steps")
        seed = gr.Number(value=default_seed, label="Seed", precision=0)
        do_sample = gr.Checkbox(value=default_do_sample, label="Do Sample")
        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=default_temperature, step=0.05, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=default_top_p, step=0.05, label="Top-p")
        top_k = gr.Number(value=default_top_k, label="Top-k", precision=0)
        force_offload = gr.Checkbox(value=default_force_offload, label="Force Offload")

    # Кнопка генерации и вывод
    generate_btn = gr.Button("🔊 Сгенерировать речь")
    output_audio = gr.Audio(label="Результат", type="filepath")

    generate_btn.click(
        fn=tts_with_save,
        inputs=[
            text, speaker_1, speaker_2, speaker_3, speaker_4,
            model_name, quantize_llm_4bit, attention_mode,
            cfg_scale, inference_steps, seed, do_sample,
            temperature, top_p, top_k, force_offload
        ],
        outputs=output_audio
    )

    load_btn.click(
        load_settings_btn,
        inputs=[],
        outputs=[text, speaker_1, speaker_2, speaker_3, speaker_4,
                model_name, quantize_llm_4bit, attention_mode,
                cfg_scale, inference_steps, seed, do_sample,
                temperature, top_p, top_k, force_offload]
    )

# ======== Разрешаем доступ к папке с аудио ========
python_dir = os.path.dirname(sys.executable)
parent_dir = os.path.abspath(os.path.join(python_dir, os.pardir))

if __name__ == "__main__":
    demo.launch(allowed_paths=[parent_dir])
