import gradio as gr
from custom_nodes.ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceTTSNode
from comfy_extras.nodes_audio import LoadAudio, SaveAudio


class VibeVoiceWorkflow:
    def __init__(self, model_name="VibeVoice-Large"):
        self.model_name = model_name
        self.tts_node = VibeVoiceTTSNode()
        self.save_node = SaveAudio()

    def run(self, text, speaker1_file, speaker2_file,
            quantize_llm_4bit=False, attention_mode="eager",
            cfg_scale=1.3, inference_steps=10, seed=1,
            do_sample=True, temperature=0.95, top_p=0.95,
            top_k=0, force_offload=False):

        # Загрузка голосов
        loadaudio_1 = LoadAudio()
        spk1 = loadaudio_1.load(audio=speaker1_file)[0]

        loadaudio_2 = LoadAudio()
        spk2 = loadaudio_2.load(audio=speaker2_file)[0]

        # Генерация речи
        result = self.tts_node.generate_audio(
            model_name=self.model_name,
            text=text,
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
            speaker_1_voice=spk1,
            speaker_2_voice=spk2
        )

        # Сохранение результата
        out_path = self.save_node.save_flac(
            filename_prefix="audio/VibeVoice",
            audio=result[0]
        )[0]

        return out_path

    def create_interface(self):
        with gr.Blocks(title="VibeVoice TTS Generator") as demo:
            gr.Markdown("## 🎙️ VibeVoice TTS Generator\nГенерация диалога с помощью VibeVoice TTS")

            with gr.Row():
                text_input = gr.Textbox(
                    label="Диалог (текст)",
                    lines=10,
                    value="Speaker 1: ...\nSpeaker 2: ..."
                )

            with gr.Row():
                spk1 = gr.Audio(label="Голос Speaker 1 (WAV)", type="filepath")
                spk2 = gr.Audio(label="Голос Speaker 2 (WAV)", type="filepath")

            with gr.Accordion("⚙️ Параметры генерации", open=False):
                seed = gr.Slider(0, 10000, value=1, step=1, label="Seed")
                temperature = gr.Slider(0.1, 1.5, value=0.95, step=0.05, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
                cfg_scale = gr.Slider(0.5, 2.0, value=1.3, step=0.1, label="CFG Scale")
                inference_steps = gr.Slider(1, 50, value=10, step=1, label="Inference Steps")
                do_sample = gr.Checkbox(value=True, label="Do Sample")
                force_offload = gr.Checkbox(value=False, label="Force Offload")
                quantize_llm_4bit = gr.Checkbox(value=False, label="Quantize LLM 4bit")

            generate_btn = gr.Button("🚀 Сгенерировать")
            output_audio = gr.Audio(label="Сгенерированное аудио")

            def generate_fn(text, spk1_file, spk2_file,
                            seed, temperature, top_p,
                            cfg_scale, inference_steps,
                            do_sample, force_offload, quantize_llm_4bit):

                return self.run(
                    text=text,
                    speaker1_file=spk1_file,
                    speaker2_file=spk2_file,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    do_sample=do_sample,
                    force_offload=force_offload,
                    quantize_llm_4bit=quantize_llm_4bit
                )

            generate_btn.click(
                fn=generate_fn,
                inputs=[text_input, spk1, spk2,
                        seed, temperature, top_p,
                        cfg_scale, inference_steps,
                        do_sample, force_offload, quantize_llm_4bit],
                outputs=output_audio
            )

        return demo


if __name__ == "__main__":
    workflow = VibeVoiceWorkflow()
    interface = workflow.create_interface()
    interface.launch()