#
# Text to Speech Converter

import torch
from scipy.io import wavfile
from pathlib import Path
from typing import Optional, Union
import logging
import gradio as gr
import os
import tempfile
from textbox_with_stt_final_online import TextboxWithSTTOnline

import base64
import mimetypes
import os
import re
import struct
from google import genai
from google.genai import types

from stress_placement_offline import StressPlacementOffline

class GeminiAudioOnline:
    def __init__(self, 
        ):
        """
        Инициализация модели преобразования текста в речь
        """

        self.render()

    examples=[
                    """Read aloud in a warm, welcoming tone
    Speaker 1: Hello! We're excited to show you our native speech capabilities
    Speaker 2: Where you can direct a voice, create realistic dialog, and so much more. Edit these placeholders to get started.""",
                    """Read aloud in a warm, welcoming tone in Russian
    Speaker 1: Ночью двадцать третьего июня начал извергаться самый высокий действующий вулкан в Евразии - Ключевской. Об этом сообщила руководитель Камчатской группы реагирования на вулканические извержения, ведущий научный сотрудник Института вулканологии и сейсмологии ДВО РАН Ольга Гирина.
    Speaker 2: Зафиксированное ночью не просто свечение, а вершинное эксплозивное извержение стромболианского типа. Пока такое извержение никому не опасно: ни населению, ни авиации
    Speaker 1: пояснила ТАСС госпожа Гирина.""",
                    """Read aloud in a warm, welcoming tone
    Speaker 1: Hello! How are you? This is a text-to-speech conversion test.""",
                    """Read aloud in a warm, welcoming tone in French
    Speaker 1: Bonjour! Comment ça va? Ceci est un test de synthèse vocale.""",
                    """Read aloud in a warm, welcoming tone in Spanish
    Speaker 1: Hola! ¿Cómo estás? Esta es una prueba de texto a voz."""
                ]

    speakers=["Zephyr -- Bright",
"Puck -- Upbeat",
"Charon -- Informative",
"Kore -- Firm",
"Fenrir -- Excitable",
"Leda -- Youthful",
"Orus -- Firm",
"Aoede -- Breezy",
"Callirrhoe -- Easy-going",
"Autonoe -- Bright",
"Enceladus -- Breathy",
"Iapetus -- Clear",
"Umbriel -- Easy-going",
"Algieba -- Smooth",
"Despina -- Smooth",
"Erinome -- Clear",
"Algenib -- Gravelly",
"Rasalgethi -- Informative",
"Laomedeia -- Upbeat",
"Achernar -- Soft",
"Alnilam -- Firm",
"Schedar -- Even",
"Gacrux -- Mature",
"Pulcherrima -- Forward",
"Achird -- Friendly",
"Zubenelgenubi -- Casual",
"Vindemiatrix -- Gentle",
"Sadachbia -- Lively",
"Sadaltager -- Knowledgeable",
"Sulafat -- Warm"]
    
    speaker1="Zephyr -- Bright"
    speaker2="Charon -- Informative"

    def save_binary_file(self, file_name, data):
        f = open(file_name, "wb")
        f.write(data)
        f.close()
        print(f"File saved to to: {file_name}")

    def parse_audio_mime_type(self, mime_type: str) -> dict[str, int | None]:
        """Parses bits per sample and rate from an audio MIME type string.

        Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

        Args:
            mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

        Returns:
            A dictionary with "bits_per_sample" and "rate" keys. Values will be
            integers if found, otherwise None.
        """
        bits_per_sample = 16
        rate = 24000

        # Extract rate from parameters
        parts = mime_type.split(";")
        for param in parts: # Skip the main type part
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    # Handle cases like "rate=" with no value or non-integer value
                    pass # Keep rate as default
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass # Keep bits_per_sample as default if conversion fails

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Generates a WAV file header for the given audio data and parameters.

        Args:
            audio_data: The raw audio data as a bytes object.
            mime_type: Mime type of the audio data.

        Returns:
            A bytes object representing the WAV file header.
        """
        parameters = self.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

        # http://soundfile.sapp.org/doc/WaveFormat/

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",          # ChunkID
            chunk_size,       # ChunkSize (total file size - 8 bytes)
            b"WAVE",          # Format
            b"fmt ",          # Subchunk1ID
            16,               # Subchunk1Size (16 for PCM)
            1,                # AudioFormat (1 for PCM)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",          # Subchunk2ID
            data_size         # Subchunk2Size (size of audio data)
        )
        return header + audio_data
        
    def generate(self, text, model="gemini-2.5-flash-preview-tts", speaker1="Zephyr -- Bright", speaker2="Charon -- Informative"):
        speaker1=speaker1.split(" -- ")[0]
        speaker2=speaker2.split(" -- ")[0]
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = model
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=[
                "audio",
            ],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 1",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=speaker1
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 2",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=speaker2
                                )
                            ),
                        ),
                    ]
                ),
            ),
        )

        file_index = 0
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                file_name = f"debug/audio_{file_index}"
                file_index += 1
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                if file_extension is None:
                    file_extension = ".wav"
                    data_buffer = self.convert_to_wav(inline_data.data, inline_data.mime_type)
                self.save_binary_file(f"{file_name}{file_extension}", data_buffer)
            else:
                print(chunk.text)

        return f"{file_name}{file_extension}"

    def render(self):            
        # Создание Gradio интерфейса
        gr.Markdown("### 🎵 Text to Speech Converter")
        #gr.Markdown("Преобразуйте текст в естественную речь с помощью AI моделей")
        
        with gr.Accordion(label="Модель", open=False):
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]),
                    value="gemini-2.5-flash-preview-tts",
                    label="Выберите модель",
                    info="Выберите модель для синтеза речи"
                )
                
        with gr.Row(scale=1):
            self.speaker1_dropdown = gr.Dropdown(
                choices=self.speakers,
                value=self.speakers[0],
                label="Выберите голос",
                info="Выберите голос для синтеза речи"
            )
            self.speaker2_dropdown = gr.Dropdown(
                choices=self.speakers,
                value=self.speakers[1],
                label="Выберите голос",
                info="Выберите голос для синтеза речи"
            )

        with gr.Column(scale=2):
            self.text_input = TextboxWithSTTOnline(
                label="Текст",
                placeholder="Введите текст для преобразования в речь...",
                lines=4,
                max_lines=10,
                value=self.examples[0]
            )

            self.stress_placement = StressPlacementOffline()
            
            generate_btn = gr.Button("🎵 Сгенерировать аудио", variant="primary")
            
            self.audio_output = gr.Audio(
                label="Сгенерированное аудио",
                type="filepath",
                interactive=True
            )
        
        generate_btn.click(
            fn=self.generate,
            inputs=[self.text_input.textbox, model_dropdown, self.speaker1_dropdown, self.speaker2_dropdown],
            outputs=[self.audio_output]
        )

        self.text_input.textbox.change(
            fn=lambda x: [x,x],
            inputs=self.text_input.textbox,
            outputs=[self.stress_placement.input_text.textbox, self.stress_placement.output_text.textbox]
        )
        
        
        # Примеры текста
        with gr.Accordion(label="Примеры", open=False):
            gr.Examples(
                examples=self.examples,
                inputs=self.text_input.textbox,
                label="Примеры текста для тестирования"
            )
        

    
    def create_interface(self):
        with gr.Blocks(title="Text to Speech Converter", theme=gr.themes.Soft()) as demo:
            self.render()

        return demo


# Запуск приложения
if __name__ == "__main__":
    with gr.Blocks(title="Text to Speech Converter", theme=gr.themes.Soft()) as demo:
        tts=GeminiAudioOnline()
    
    # Запуск интерфейса
    demo.launch(
        share=False,
        show_error=True
    )