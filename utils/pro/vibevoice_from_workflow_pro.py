import gradio as gr
from custom_nodes.ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceTTSNode
from comfy_extras.nodes_audio import LoadAudio, SaveAudio
import os
import sys
from textbox_with_stt_final_pro import TextboxWithSTTPro


class VibeVoiceWorkflowPro:
    def __init__(self, model_name="VibeVoice-1.5B"):
        self.model_name = model_name
        #self.tts_node = VibeVoiceTTSNode()
        self.save_node = SaveAudio()
        self.create_interface()

    def run(self, text, speaker1_file, speaker2_file, speaker3_file, speaker4_file,
            quantize_llm_4bit=False, attention_mode="eager",
            cfg_scale=1.3, inference_steps=10, seed=1,
            do_sample=True, temperature=0.95, top_p=0.95,
            top_k=0, force_offload=False):

        # Загрузка голосов
        loadaudio_1 = LoadAudio()
        spk1 = loadaudio_1.load(audio=speaker1_file)[0]
        loadaudio_2 = LoadAudio()
        spk2 = loadaudio_2.load(audio=speaker2_file)[0]
        loadaudio_3 = LoadAudio()
        spk3 = loadaudio_3.load(audio=speaker3_file)[0]
        loadaudio_4 = LoadAudio()
        spk4 = loadaudio_4.load(audio=speaker4_file)[0]

        tts_node = VibeVoiceTTSNode()
        # Генерация речи
        result = tts_node.generate_audio(
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
            speaker_2_voice=spk2,
            speaker_3_voice=spk3,
            speaker_4_voice=spk4
        )
        tts_node.cpu()
        del tts_node
        tts_node=0
        import torch
        torch.cuda.empty_cache()
        # Сохранение результата
        out_path = self.save_node.save_flac(
            filename_prefix="audio/VibeVoice",
            audio=result[0]
        )

        comfyui_dir = os.path.join(self.allowed_paths, "ComfyUI")
        full_path = f"{comfyui_dir}/{out_path['ui']['audio'][0]["type"]}/{out_path['ui']['audio'][0]["subfolder"]}/{out_path['ui']['audio'][0]["filename"]}"

        return full_path

    def create_interface(self):
        gr.Markdown("### 🎙️ VibeVoice TTS Generator\nГенерация диалога с помощью VibeVoice TTS")

        python_dir = os.path.dirname(sys.executable)
        self.allowed_paths = os.path.abspath(os.path.join(python_dir, os.pardir))
        vibe_models = gr.Dropdown(label="Модель", choices=["VibeVoice-1.5B", "VibeVoice-Large"], value="VibeVoice-1.5B")
        
        def update_str(value):
            self.model_name=value

        vibe_models.change(fn=update_str,
                           inputs=vibe_models,
                           outputs=None)
        with gr.Row():
            self.text_input = TextboxWithSTTPro(
                label="Диалог (текст)",
                lines=10,
                value="Speaker 1: Hi! Good morning!\nSpeaker 2: How are you?"
            )

        with gr.Accordion("⚙️ Голоса", open=False):
            with gr.Row():
                spk1 = gr.Audio(label="Голос Speaker 1 (WAV)", type="filepath", value="vibevoice/voices/ru-1.wav")
                spk2 = gr.Audio(label="Голос Speaker 2 (WAV)", type="filepath", value="vibevoice/voices/ru-1.wav")
                spk3 = gr.Audio(label="Голос Speaker 3 (WAV)", type="filepath", value="vibevoice/voices/ru-1.wav")
                spk4 = gr.Audio(label="Голос Speaker 4 (WAV)", type="filepath", value="vibevoice/voices/ru-1.wav")

        with gr.Accordion("⚙️ Параметры генерации", open=False):
            seed = gr.Number(label="Seed", value=40)
            temperature = gr.Slider(0.1, 1.5, value=0.95, step=0.05, label="Temperature")
            top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
            cfg_scale = gr.Slider(0.5, 2.0, value=1.3, step=0.1, label="CFG Scale")
            inference_steps = gr.Slider(1, 50, value=10, step=1, label="Inference Steps")
            do_sample = gr.Checkbox(value=True, label="Do Sample")
            force_offload = gr.Checkbox(value=False, label="Force Offload")
            quantize_llm_4bit = gr.Checkbox(value=False, label="Quantize LLM 4bit")

        generate_btn = gr.Button("🚀 Сгенерировать")
        self.output_audio = gr.Audio(label="Сгенерированное аудио", interactive=True, type="filepath")

        def generate_fn(text, spk1_file, spk2_file, spk3_file, spk4_file,
                        seed, temperature, top_p,
                        cfg_scale, inference_steps,
                        do_sample, force_offload, quantize_llm_4bit):

            return self.run(
                text=text,
                speaker1_file=spk1_file,
                speaker2_file=spk2_file,
                speaker3_file=spk3_file,
                speaker4_file=spk4_file,
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
            inputs=[self.text_input.textbox, spk1, spk2, spk3, spk4,
                    seed, temperature, top_p,
                    cfg_scale, inference_steps,
                    do_sample, force_offload, quantize_llm_4bit],
            outputs=self.output_audio
        )




if __name__ == "__main__":
    with gr.Blocks(title="VibeVoice TTS Generator") as demo: 
        workflow = VibeVoiceWorkflowPro()

    demo.launch(allowed_paths=[workflow.allowed_paths])