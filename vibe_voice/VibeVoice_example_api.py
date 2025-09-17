"""
Auto-generated ComfyUI workflow Python script
Generated from Export (API) JSON format

Коррекция: 
- папка была уже изменена (изменено в названии папки "- 2 на "_"), 
- добавлен "custom_nodes.", папка была уже изменена(изменено "- 2 на "_" в "from ..."), 
- добавлен "\" в промпте в переносах строк, можно вместо этого использовать три кавычки вместо '"' 
- удалено 'audioUI=""'
"""

from custom_nodes.ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceTTSNode
from comfy_extras.nodes_audio import LoadAudio, SaveAudio
import os

def run_vibevoice_tts(
    text: str,
    speaker_1_path: str,
    speaker_2_path: str,
    speaker_3_path: str,
    speaker_4_path: str,
    model_name: str = "VibeVoice-Large",
    quantize_llm_4bit: bool = False,
    attention_mode: str = "eager",
    cfg_scale: float = 1.3,
    inference_steps: int = 10,
    seed: int = 1,
    do_sample: bool = True,
    temperature: float = 0.95,
    top_p: float = 0.95,
    top_k: int = 0,
    force_offload: bool = False,
    output_prefix: str = "audio/VibeVoice"
):
    """Запуск генерации речи через VibeVoice"""

    # Загружаем голоса
    loadaudio = LoadAudio()
    speaker_1 = loadaudio.load(audio=speaker_1_path)[0] if speaker_1_path else None
    speaker_2 = loadaudio.load(audio=speaker_2_path)[0] if speaker_2_path else None
    speaker_3 = loadaudio.load(audio=speaker_3_path)[0] if speaker_3_path else None
    speaker_4 = loadaudio.load(audio=speaker_4_path)[0] if speaker_4_path else None

    # Генерация речи
    vibevoicettsnode = VibeVoiceTTSNode()
    output = vibevoicettsnode.generate_audio(
        model_name=model_name,
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
        speaker_1_voice=speaker_1,
        speaker_2_voice=speaker_2,
        speaker_3_voice=speaker_3,
        speaker_4_voice=speaker_4
    )

    # Сохраняем результат
    saveaudio = SaveAudio()
    result = saveaudio.save_flac(filename_prefix=output_prefix, audio=output[0])

    # Путь из структуры результата
    filename = result["ui"]["audio"][0]["filename"]
    subfolder = result["ui"]["audio"][0]["subfolder"]

    # Абсолютный путь строим от saveaudio.output_dir
    abs_path = os.path.join(saveaudio.output_dir, subfolder, filename)
    abs_path = os.path.abspath(abs_path)

    return abs_path  # путь к файлу


if __name__ == "__main__":
    # Пример запуска
    path = run_vibevoice_tts(
        text="""Speaker 1: Не могу поверить, что ты снова это сделал. Я ждал два часа. Два часа! Ни одного звонка, ни одного сообщения. Ты хоть представляешь, как это было неловко — просто сидеть там в одиночестве?
Speaker 2: Слушай, я знаю, извини, ладно? Работа была настоящим кошмаром. Мой начальник в последнюю минуту поставил мне критически важный дедлайн. У меня даже секунды не было, чтобы вздохнуть, не говоря уже о том, чтобы проверить телефон.
Speaker 1: Кошмар? Это же оправдание ты использовал в прошлый раз. Я начинаю думать, что тебе просто всё равно. Легче сказать «работа была сумасшедшей», чем признать, что я больше не в приоритете для тебя.""",
        speaker_1_path="1.wav",
        speaker_2_path="5.wav",
        speaker_3_path="5.wav",
        speaker_4_path="5.wav"
    )
    print("Сохранено в:", path)