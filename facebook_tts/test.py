from transformers import VitsModel, AutoTokenizer
import torch
import scipy

device = 'cuda' #  'cpu' or 'cuda'

speaker = 1 # 0-woman, 1-man  

# load model
model_name = "utrobinmv/tts_ru_free_hf_vits_high_multispeaker"

model = VitsModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# text with accents
text = """Ночью двадцать тр+етьего июня начал извергаться самый высокий 
действующий вулк+ан в Евразии - Кл+ючевской. Об этом сообщила руководитель 
Камчатской группы реагирования на вулканические извержения, ведущий 
научный сотрудник Института вулканологии и сейсмологии ДВО РАН +Ольга Гирина.
«Зафиксированное ночью не просто свечение, а вершинное эксплозивное 
извержение стромболианского типа. Пока такое извержение никому не опасно: 
ни населению, ни авиации» пояснила ТАСС госпожа Гирина."""

# text lowercase
text = text.lower()

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs.to(device), speaker_id=speaker).waveform
    output = output.detach().cpu().numpy()
    
scipy.io.wavfile.write("tts_audio.wav", rate=model.config.sampling_rate,
                       data=output[0])
