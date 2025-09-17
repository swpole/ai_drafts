import gradio as gr
from VibeVoice_example_api import run_vibevoice_tts
import sys, os

def tts_interface(
    text,
    speaker_1,
    speaker_2,
    speaker_3,
    speaker_4,
    model_name,
    quantize_llm_4bit,
    attention_mode,
    cfg_scale,
    inference_steps,
    seed,
    do_sample,
    temperature,
    top_p,
    top_k,
    force_offload
):
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
        force_offload=force_offload,
    )
    return result


with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ VibeVoice TTS Demo")

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ VibeVoice TTS Demo")

    default_speaker_1 = r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac"
    default_speaker_2 = r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac"
    default_speaker_3 = r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac"
    default_speaker_4 = r"D:\\ComfyUI_portable\\ComfyUI_windows_portable\\ComfyUI\\output\\audio\\VibeVoice_00002_.flac"

    with gr.Row():
        # Голос 1
        with gr.Column(scale=0.25):
            speaker_1 = gr.File(label="Голос 1", type="filepath", value=default_speaker_1)
            preview_1 = gr.Audio(label="▶", type="filepath", value=default_speaker_1)
            speaker_1.change(lambda f: f.name if f else None, inputs=speaker_1, outputs=preview_1)

        # Голос 2
        with gr.Column(scale=0.25):
            speaker_2 = gr.File(label="Голос 2", type="filepath", value=default_speaker_2)
            preview_2 = gr.Audio(label="▶", type="filepath", value=default_speaker_2)
            speaker_2.change(lambda f: f.name if f else None, inputs=speaker_2, outputs=preview_2)

        # Голос 3
        with gr.Column(scale=0.25):
            speaker_3 = gr.File(label="Голос 3", type="filepath", value=default_speaker_3)
            preview_3 = gr.Audio(label="▶", type="filepath", value=default_speaker_3)
            speaker_3.change(lambda f: f.name if f else None, inputs=speaker_3, outputs=preview_3)

        # Голос 4
        with gr.Column(scale=0.25):
            speaker_4 = gr.File(label="Голос 4", type="filepath", value=default_speaker_4)
            preview_4 = gr.Audio(label="▶", type="filepath", value=default_speaker_4)
            speaker_4.change(lambda f: f.name if f else None, inputs=speaker_4, outputs=preview_4)


    text = gr.Textbox(label="Введите текст", lines=6, placeholder="Напишите здесь текст для озвучки...")

    with gr.Accordion("Дополнительные параметры", open=False):
        model_name = gr.Textbox(value="VibeVoice-Large", label="Модель")
        quantize_llm_4bit = gr.Checkbox(value=False, label="Quantize LLM 4bit")
        attention_mode = gr.Dropdown(choices=["eager", "flash"], value="eager", label="Attention mode")
        cfg_scale = gr.Slider(minimum=0.1, maximum=3.0, value=1.3, step=0.1, label="CFG Scale")
        inference_steps = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Inference steps")
        seed = gr.Number(value=1, label="Seed", precision=0)
        do_sample = gr.Checkbox(value=True, label="Do Sample")
        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.95, step=0.05, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")
        top_k = gr.Number(value=0, label="Top-k", precision=0)
        force_offload = gr.Checkbox(value=False, label="Force Offload")

    generate_btn = gr.Button("🔊 Сгенерировать речь")
    output_audio = gr.Audio(label="Результат", type="filepath")

    generate_btn.click(
        fn=tts_interface,
        inputs=[
            text,
            speaker_1, speaker_2, speaker_3, speaker_4,
            model_name, quantize_llm_4bit, attention_mode,
            cfg_scale, inference_steps, seed, do_sample,
            temperature, top_p, top_k, force_offload
        ],
        outputs=output_audio
    )

if __name__ == "__main__":
    python_dir = os.path.dirname(sys.executable)
    parent_dir = os.path.abspath(os.path.join(python_dir, os.pardir))

    demo.launch(
        allowed_paths=[parent_dir]
    )
