import torch
from scipy.io import wavfile
from transformers import VitsModel, AutoTokenizer

model_name = "utrobinmv/tts_ru_free_hf_vits_high_multispeaker"
# model_name = "facebook/mms-tts-rus"
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "У меня есть код, который преобразует текст в речь. Как сохранить результат в файл. Вот этот код:"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs, speaker_id=0).waveform

# Преобразуем тензор в numpy массив и нормализуем
audio = output.cpu().numpy()
audio = audio.squeeze()  # Убираем лишние измерения
audio = (audio * 32767).astype('int16')  # Нормализуем для 16-битного WAV

# Сохраняем в WAV файл
sample_rate = model.config.sampling_rate
wavfile.write("output_audio.wav", sample_rate, audio)

print("Аудио сохранено в файл: output_audio.wav")