import gc
import json
import os
import re
import tempfile
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    tempfile_kwargs,
)
from f5_tts.model import DiT, UNetT


class F5TTSApp:
    def __init__(self):
        self.DEFAULT_TTS_MODEL = "F5-TTS_v1"
        self.tts_model_choice = self.DEFAULT_TTS_MODEL
        
        self.DEFAULT_TTS_MODEL_CFG = [
            "hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/model_240000_inference.safetensors",
            "hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt",
            json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
        ]
        
        # Load models
        self.vocoder = load_vocoder()
        self.F5TTS_ema_model = self.load_f5tts()
        self.E2TTS_ema_model = None  # Для локального окружения не загружаем E2-TTS по умолчанию
        self.custom_ema_model = None
        self.pre_custom_path = ""
        
        self.chat_model_state = None
        self.chat_tokenizer_state = None
        
        self.app = self.create_app()

    def load_f5tts(self):
        ckpt_path = str(cached_path(self.DEFAULT_TTS_MODEL_CFG[0]))
        F5TTS_model_cfg = json.loads(self.DEFAULT_TTS_MODEL_CFG[2])
        return load_model(DiT, F5TTS_model_cfg, ckpt_path)

    def load_e2tts(self):
        ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
        E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
        return load_model(UNetT, E2TTS_model_cfg, ckpt_path)

    def load_custom(self, ckpt_path: str, vocab_path="", model_cfg=None):
        ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
        if ckpt_path.startswith("hf://"):
            ckpt_path = str(cached_path(ckpt_path))
        if vocab_path.startswith("hf://"):
            vocab_path = str(cached_path(vocab_path))
        if model_cfg is None:
            model_cfg = json.loads(self.DEFAULT_TTS_MODEL_CFG[2])
        elif isinstance(model_cfg, str):
            model_cfg = json.loads(model_cfg)
        return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

    def load_text_from_file(self, file):
        if file:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = ""
        return gr.update(value=text)

    @lru_cache(maxsize=1000)
    def infer(
        self,
        ref_audio_orig,
        ref_text,
        gen_text,
        model,
        remove_silence,
        seed,
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1,
        show_info=gr.Info,
    ):
        if not ref_audio_orig:
            gr.Warning("Please provide reference audio.")
            return gr.update(), gr.update(), ref_text

        # Set inference seed
        if seed < 0 or seed > 2**31 - 1:
            gr.Warning("Seed must in range 0 ~ 2147483647. Using random seed instead.")
            seed = np.random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        used_seed = seed

        if not gen_text.strip():
            gr.Warning("Please enter text to generate or upload a text file.")
            return gr.update(), gr.update(), ref_text

        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

        if model == self.DEFAULT_TTS_MODEL:
            ema_model = self.F5TTS_ema_model
        elif model == "E2-TTS":
            if self.E2TTS_ema_model is None:
                show_info("Loading E2-TTS model...")
                self.E2TTS_ema_model = self.load_e2tts()
            ema_model = self.E2TTS_ema_model
        elif isinstance(model, tuple) and model[0] == "Custom":
            if self.pre_custom_path != model[1]:
                show_info("Loading Custom TTS model...")
                self.custom_ema_model = self.load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
                self.pre_custom_path = model[1]
            ema_model = self.custom_ema_model

        final_wave, final_sample_rate = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            self.vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=show_info,
            progress=gr.Progress(),
        )

        # Remove silence
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
                temp_path = f.name
            try:
                sf.write(temp_path, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            finally:
                os.unlink(temp_path)
            final_wave = final_wave.squeeze().cpu().numpy()

        return (final_sample_rate, final_wave), ref_text, used_seed

    def load_last_used_custom(self):
        last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return self.DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(self, new_choice):
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = self.load_last_used_custom()
            self.tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            self.tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(self, custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        self.tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
        last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    def basic_tts(
        self,
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        randomize_seed,
        seed_input,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        if randomize_seed:
            seed_input = np.random.randint(0, 2**31 - 1)

        audio_out, ref_text_out, used_seed = self.infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            self.tts_model_choice,
            remove_silence,
            seed=seed_input,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, ref_text_out, used_seed

    def create_app(self):
        with gr.Blocks() as app:
            gr.Markdown(f"""### F5-TTS Demo Space""")

            with gr.Row():
                choose_tts_model = gr.Radio(
                    choices=[self.DEFAULT_TTS_MODEL, "E2-TTS", "Custom"], 
                    label="Choose TTS Model", 
                    value=self.DEFAULT_TTS_MODEL
                )
                custom_ckpt_path = gr.Dropdown(
                    choices=[self.DEFAULT_TTS_MODEL_CFG[0]],
                    value=self.load_last_used_custom()[0],
                    allow_custom_value=True,
                    label="Model: local_path | hf://user_id/repo_id/model_ckpt",
                    visible=False,
                )
                custom_vocab_path = gr.Dropdown(
                    choices=[self.DEFAULT_TTS_MODEL_CFG[1]],
                    value=self.load_last_used_custom()[1],
                    allow_custom_value=True,
                    label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
                    visible=False,
                )
                custom_model_cfg = gr.Dropdown(
                    choices=[
                        self.DEFAULT_TTS_MODEL_CFG[2],
                        json.dumps(
                            dict(
                                dim=1024,
                                depth=22,
                                heads=16,
                                ff_mult=2,
                                text_dim=512,
                                text_mask_padding=False,
                                conv_layers=4,
                                pe_attn_head=1,
                            )
                        ),
                        json.dumps(
                            dict(
                                dim=768,
                                depth=18,
                                heads=12,
                                ff_mult=2,
                                text_dim=512,
                                text_mask_padding=False,
                                conv_layers=4,
                                pe_attn_head=1,
                            )
                        ),
                    ],
                    value=self.load_last_used_custom()[2],
                    allow_custom_value=True,
                    label="Config: in a dictionary form",
                    visible=False,
                )

            choose_tts_model.change(
                self.switch_tts_model,
                inputs=[choose_tts_model],
                outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
                show_progress="hidden",
            )
            custom_ckpt_path.change(
                self.set_custom_model,
                inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
                show_progress="hidden",
            )
            custom_vocab_path.change(
                self.set_custom_model,
                inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
                show_progress="hidden",
            )
            custom_model_cfg.change(
                self.set_custom_model,
                inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
                show_progress="hidden",
            )

            ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Row():
                gen_text_input = gr.Textbox(
                    label="Text to Generate",
                    lines=10,
                    max_lines=40,
                    scale=4,
                )

            generate_btn = gr.Button("Synthesize", variant="primary")
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    ref_text_input = gr.Textbox(
                        label="Reference Text",
                        info="Leave blank to automatically transcribe the reference audio. If you enter text or upload a file, it will override automatic transcription.",
                        lines=2,
                        scale=4,
                    )
                    ref_text_file = gr.File(label="Load Reference Text from File (.txt)", file_types=[".txt"], scale=1)
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        info="Check to use a random seed for each generation. Uncheck to use the seed specified.",
                        value=True,
                        scale=3,
                    )
                    seed_input = gr.Number(show_label=False, value=0, precision=0, scale=1)
                    with gr.Column(scale=4):
                        remove_silence = gr.Checkbox(
                            label="Remove Silences",
                            info="If undesired long silence(s) produced, turn on to automatically detect and crop.",
                            value=False,
                        )
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.3,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    info="Adjust the speed of the audio.",
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4,
                    maximum=64,
                    value=32,
                    step=2,
                    info="Set the number of denoising steps.",
                )
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (s)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    info="Set the duration of the cross-fade between audio clips.",
                )

            audio_output = gr.Audio(label="Synthesized Audio")

            ref_text_file.upload(
                self.load_text_from_file,
                inputs=[ref_text_file],
                outputs=[ref_text_input],
            )

            ref_audio_input.clear(
                lambda: [None, None],
                None,
                [ref_text_input, ref_text_file],
            )

            generate_btn.click(
                self.basic_tts,
                inputs=[
                    ref_audio_input,
                    ref_text_input,
                    gen_text_input,
                    remove_silence,
                    randomize_seed,
                    seed_input,
                    cross_fade_duration_slider,
                    nfe_slider,
                    speed_slider,
                ],
                outputs=[audio_output, ref_text_input, seed_input],
            )

        return app

    def launch(self, port=None, host=None, share=False, api=True, root_path=None, inbrowser=False):
        print("Starting F5-TTS app...")
        self.app.queue(api_open=api).launch(
            server_name=host,
            server_port=port,
            share=share,
            show_api=api,
            root_path=root_path,
            inbrowser=inbrowser,
        )


if __name__ == "__main__":
    app = F5TTSApp()
    app.launch()