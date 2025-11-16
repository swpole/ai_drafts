"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed
from textbox_with_stt_final_pro import TextboxWithSTTPro

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemoPro:
    def __init__(self):
        """Initialize the VibeVoice demo with model loading."""
        # model_path = "aoi-ot/VibeVoice-Large"
        model_path = "aoi-ot/VibeVoice-1.5B"
        device = "cuda"
        inference_steps = 5
        self.seed = 42
        set_seed(self.seed)  # Set a fixed seed for reproducibility
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        #self.load_model(self.model_path)
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts
        self.create_demo_interface()

        
    def load_model(self, model_path):
        """Load the VibeVoice model and processor."""
        self.model_path = model_path
        print(f"Loading processor & model from {self.model_path}")
        # Normalize potential 'mpx'
        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        print(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"[ERROR] : {type(e).__name__}: {e}")
                print(traceback.format_exc())
                fallback_attn = "sdpa"
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation=fallback_attn,
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        self.model.eval()
        
        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "vibevoice/voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices directory.")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3) -> Iterator[tuple]:
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Move tensors to device
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)
    

    def create_demo_interface(self):
        """Create the Gradio interface with streaming support."""
        gr.Markdown("### 🎙️ Vibe Podcasting")
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1):
                models = gr.Dropdown(choices=["aoi-ot/VibeVoice-Large", "aoi-ot/VibeVoice-1.5B"], label="Model", value=self.model_path)           
                with gr.Accordion("Settings", open=False):
                    seed=gr.Number(label="Seed", value=self.seed)
                    seed.change(
                        fn=set_seed,
                        inputs=[seed],
                    )

                    # Number of speakers
                    num_speakers = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Number of Speakers",
                    )

                    models.change(
                        fn=self.load_model,
                        inputs=[models],
                    )
                
                    # Speaker selection
                    gr.Markdown("### 🎭 **Speaker Selection**")
                
                    available_speaker_names = list(self.available_voices.keys())
                    # default_speakers = available_speaker_names[:4] if len(available_speaker_names) >= 4 else available_speaker_names
                    default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                    speaker_selections = []
                    for i in range(4):
                        default_value = default_speakers[i] if i < len(default_speakers) else None
                        speaker = gr.Dropdown(
                            choices=available_speaker_names,
                            value=default_value,
                            label=f"Speaker {i+1}",
                            visible=(i < 2),  # Initially show only first 2 speakers
                        )
                        speaker_selections.append(speaker)
                
                    # Advanced settings
                    gr.Markdown("### ⚙️ **Advanced Settings**")
                
                    # Sampling parameters (contains all generation settings)
                    with gr.Accordion("Generation Parameters", open=False):
                        cfg_scale = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.3,
                            step=0.05,
                            label="CFG Scale (Guidance Strength)",
                        )
                
            # Right column - Generation
            with gr.Column(scale=2, ):
                self.text_input = TextboxWithSTTPro(
                    label="Text",
                    placeholder="""Enter your podcast script here. You can format it as:

Speaker 1: Welcome to our podcast today!
Speaker 2: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                    lines=12,
                    max_lines=20,
                )
                
                with gr.Row():
                    # Generate button
                    generate_btn = gr.Button(
                        "🚀 Generate Podcast",
                        size="lg",
                        variant="primary",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                self.complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download after generation)",
                    type="numpy",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=True,  # Initially hidden, shown when audio is ready
                    interactive=True
                )

                with gr.Accordion("Help", open=False):
                    # Button row with Random Example
                    with gr.Row():
                        # Random example button (now on the left)
                        random_example_btn = gr.Button(
                            "🎲 Random Example",
                            size="lg",
                            variant="secondary",
                            scale=1  # Smaller width
                        )
                    
                    gr.Markdown("""
                    *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                    *💡 **Complete Audio**: Will appear below after generation finishes*
                    """)
                    
                    # Generation log
                    log_output = gr.Textbox(
                        label="Generation Log",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                    )
        
                    def update_speaker_visibility(num_speakers):
                        updates = []
                        for i in range(4):
                            updates.append(gr.update(visible=(i < num_speakers)))
                        return updates
                    
                    num_speakers.change(
                        fn=update_speaker_visibility,
                        inputs=[num_speakers],
                        outputs=speaker_selections
                    )
                    
                    # Main generation function with streaming
                    def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
                        """Wrapper function to handle the streaming generation call."""
                        self.load_model(self.model_path)
                        try:
                            # Extract speakers and parameters
                            speakers = speakers_and_params[:4]  # First 4 are speaker selections
                            cfg_scale = speakers_and_params[4]   # CFG scale
                            
                            # Clear outputs and reset visibility at start
                            yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                            
                            # The generator will yield multiple times
                            final_log = "Starting generation..."
                            
                            for streaming_audio, complete_audio, log, streaming_visible in self.generate_podcast_streaming(
                                num_speakers=int(num_speakers),
                                script=script,
                                speaker_1=speakers[0],
                                speaker_2=speakers[1],
                                speaker_3=speakers[2],
                                speaker_4=speakers[3],
                                cfg_scale=cfg_scale
                            ):
                                final_log = log
                                
                                # Check if we have complete audio (final yield)
                                if complete_audio is not None:
                                    # Final state: clear streaming, show complete audio
                                    yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                                else:
                                    # Streaming state: update streaming audio only
                                    if streaming_audio is not None:
                                        yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                                    else:
                                        # No new audio, just update status
                                        yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

                        except Exception as e:
                            error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                            print(error_msg)
                            import traceback
                            traceback.print_exc()
                            # Reset button states on error
                            yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    
                    def stop_generation_handler():
                        """Handle stopping generation."""
                        self.stop_audio_generation()
                        # Return values for: log_output, streaming_status, generate_btn, stop_btn
                        return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    
                    # Add a clear audio function
                    def clear_audio_outputs():
                        """Clear both audio outputs before starting new generation."""
                        return None, gr.update(value=None, visible=False)

                    # Connect generation button with streaming outputs
                    generate_btn.click(
                        fn=clear_audio_outputs,
                        inputs=[],
                        outputs=[audio_output, self.complete_audio_output],
                        queue=False
                    ).then(  # Immediate UI update to hide Generate, show Stop (non-queued)
                        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                        inputs=[],
                        outputs=[generate_btn, stop_btn],
                        queue=False
                    ).then(
                        fn=generate_podcast_wrapper,
                        inputs=[num_speakers, self.text_input.textbox] + speaker_selections + [cfg_scale],
                        outputs=[audio_output, self.complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
                        queue=True  # Enable Gradio's built-in queue
                    )
                    
                    # Connect stop button
                    stop_btn.click(
                        fn=stop_generation_handler,
                        inputs=[],
                        outputs=[log_output, streaming_status, generate_btn, stop_btn],
                        queue=False  # Don't queue stop requests
                    ).then(
                        # Clear both audio outputs after stopping
                        fn=lambda: (None, None),
                        inputs=[],
                        outputs=[audio_output, self.complete_audio_output],
                        queue=False
                    )
                    
                    # Function to randomly select an example
                    def load_random_example():
                        """Randomly select and load an example script."""
                        import random
                        
                        # Get available examples
                        if hasattr(self, 'example_scripts') and self.example_scripts:
                            example_scripts = self.example_scripts
                        else:
                            # Fallback to default
                            example_scripts = [
                                [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                            ]
                        
                        # Randomly select one
                        if example_scripts:
                            selected = random.choice(example_scripts)
                            num_speakers_value = selected[0]
                            script_value = selected[1]
                            
                            # Return the values to update the UI
                            return num_speakers_value, script_value
                        
                        # Default values if no examples
                        return 2, ""
                    
                    # Connect random example button
                    random_example_btn.click(
                        fn=load_random_example,
                        inputs=[],
                        outputs=[num_speakers, self.text_input.textbox],
                        queue=False  # Don't queue this simple operation
                    )
                    
                    # Add usage tips
                    gr.Markdown("""
                    ### 💡 **Usage Tips**
                    
                    - Click **🚀 Generate Podcast** to start audio generation
                    - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
                    - **Complete Audio** tab provides the full, uninterrupted podcast after generation
                    - During generation, you can click **🛑 Stop Generation** to interrupt the process
                    - The streaming indicator shows real-time generation progress
                    """)
        
                    # Add example scripts
                    gr.Markdown("### 📚 **Example Scripts**")
                    
                    # Use dynamically loaded examples if available, otherwise provide a default
                    if hasattr(self, 'example_scripts') and self.example_scripts:
                        example_scripts = self.example_scripts
                    else:
                        # Fallback to a simple default example if no scripts loaded
                        example_scripts = [
                            [1, "Speaker 1: Welcome to our AI podcast demonstration! This is a sample script showing how VibeVoice can generate natural-sounding speech."]
                        ]
                    
                    gr.Examples(
                        examples=example_scripts,
                        inputs=[num_speakers, self.text_input.textbox],
                        label="Try these example scripts:"
                    )

                    # --- Risks & limitations (footer) ---
                    gr.Markdown(
                        """
            ## Risks and limitations

            While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
            Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.
                        """,
                    )
        return 


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def main():
    """Main function to run the demo."""

    # Initialize demo instance
    
    
    
    with gr.Blocks(
            title="VibeVoice - AI Podcast Generator",
            #css=demo_instance.custom_css,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="purple",
                neutral_hue="slate",
            )
        ) as interface:
        demo_instance = VibeVoiceDemoPro()
    interface.launch()


if __name__ == "__main__":
    main()
