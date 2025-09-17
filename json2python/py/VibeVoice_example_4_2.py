"""
Auto-generated ComfyUI workflow Python script
Generated from Export (API) JSON format
"""

from ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceTTSNode
from comfy_extras.nodes_audio import LoadAudio
from comfy_extras.nodes_audio import SaveAudio
from comfy_extras.nodes_primitive import String
from nodes import *
import folder_paths
import os
import sys
import torch

def main():
    """Основная функция выполнения workflow"""

    loadaudio_4 = LoadAudio()
    loadaudio_4_output = loadaudio_4.load(audio="1.wav", audioUI="")
    # Output available as loadaudio_4_output
    
    loadaudio_8 = LoadAudio()
    loadaudio_8_output = loadaudio_8.load(audio="5.wav", audioUI="")
    # Output available as loadaudio_8_output
    
    loadaudio_12 = LoadAudio()
    loadaudio_12_output = loadaudio_12.load(audio="5.wav", audioUI="")
    # Output available as loadaudio_12_output
    
    loadaudio_13 = LoadAudio()
    loadaudio_13_output = loadaudio_13.load(audio="5.wav", audioUI="")
    # Output available as loadaudio_13_output
    
    string_15 = String()
    string_15_output = string_15.execute(value="Speaker 1: Не могу поверить, что ты снова это сделал. Я ждал два часа. Два часа! Ни одного звонка, ни одного сообщения. Ты хоть представляешь, как это было неловко — просто сидеть там в одиночестве? Speaker 2: Слушай, я знаю, извини, ладно? Работа была настоящим кошмаром. Мой начальник в последнюю минуту поставил мне критически важный дедлайн. У меня даже секунды не было, чтобы вздохнуть, не говоря уже о том, чтобы проверить телефон. Speaker 1: Кошмар? Это же оправдание ты использовал в прошлый раз. Я начинаю думать, что тебе просто всё равно. Легче сказать «работа была сумасшедшей», чем признать, что я больше не в приоритете для тебя.")
    # Output available as string_15_output
    
    vibevoicettsnode_11 = VibeVoiceTTSNode()
    vibevoicettsnode_11_output = vibevoicettsnode_11.generate_audio(model_name="VibeVoice-Large", text=string_15_output[0], quantize_llm_4bit=False, attention_mode="eager", cfg_scale=1.3, inference_steps=10, seed=321335798495663, do_sample=True, temperature=0.95, top_p=0.95, top_k=0, force_offload=False, speaker_1_voice=loadaudio_4_output[0], speaker_2_voice=loadaudio_8_output[0], speaker_3_voice=loadaudio_12_output[0], speaker_4_voice=loadaudio_13_output[0])
    # Output available as vibevoicettsnode_11_output
    
    saveaudio_3 = SaveAudio()
    saveaudio_3_output = saveaudio_3.save_flac(filename_prefix="audio/VibeVoice", audioUI="", audio=vibevoicettsnode_11_output[0])
    # Output available as saveaudio_3_output
    

if __name__ == "__main__":
    main()