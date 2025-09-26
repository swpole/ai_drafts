import gradio as gr
import ollama
import os
import tempfile
import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer
import numpy as np
from scipy.io import wavfile
import io
import soundfile as sf
from typing import List, Dict, Tuple, Optional
import json
import pickle

# Для голосового ввода
import speech_recognition as sr
from pydub import AudioSegment

class VitsTTS:
    def __init__(self, model_name: str = "facebook/mms-tts-rus"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели TTS"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
            print(f"Модель TTS {self.model_name} загружена на устройство: {self.device}")
        except Exception as e:
            print(f"Ошибка загрузки модели TTS: {e}")
    
    def text_to_speech(self, text: str, speed: float = 1.0) -> str:
        """Преобразование текста в речь с помощью VitsModel"""
        if not text.strip():
            return ""
        
        try:
            # Токенизация текста
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Генерация аудио
            with torch.no_grad():
                output = self.model(**inputs)
            
            # Извлечение аудио данных
            audio_data = output.waveform.cpu().numpy()[0]
            sample_rate = self.model.config.sampling_rate
            
            # Изменение скорости воспроизведения (опционально)
            if speed != 1.0:
                audio_data = self._change_speed(audio_data, sample_rate, speed)
            
            # Сохранение во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                return tmp_file.name
                
        except Exception as e:
            print(f"Ошибка синтеза речи: {e}")
            return ""
    
    def _change_speed(self, audio_data: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
        """Изменение скорости аудио"""
        if speed == 1.0:
            return audio_data
            
        # Простое изменение скорости через ресемплинг
        new_length = int(len(audio_data) / speed)
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)

class RAGChatSystem:
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.conversation_history = []
        self.vector_db = []
        
    def add_to_knowledge_base(self, documents: List[str]):
        """Добавление документов в базу знаний"""
        for doc in documents:
            self.vector_db.append({
                "content": doc,
                "embedding": self.get_embedding(doc)
            })
    
    def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга текста"""
        try:
            # Используем Ollama для получения эмбеддингов
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            return response['embedding']
        except:
            # Запасной вариант
            return [len(text)] * 384
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[str]:
        """Семантический поиск в базе знаний"""
        if not self.vector_db:
            return []
            
        query_embedding = self.get_embedding(query)
        
        results = []
        for doc in self.vector_db:
            similarity = self.cosine_similarity(query_embedding, doc["embedding"])
            results.append((similarity, doc["content"]))
        
        results.sort(reverse=True)
        return [content for _, content in results[:top_k]]
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Вычисление косинусной схожести"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def generate_response(self, user_input: str) -> Tuple[str, str]:
        """Генерация ответа с использованием RAG"""
        # Поиск релевантной информации
        relevant_docs = self.semantic_search(user_input)
        context = "\n".join(relevant_docs) if relevant_docs else "Нет релевантной информации в базе знаний."
        
        # Формирование промпта с контекстом
        prompt = f"""Контекст для ответа:
{context}

Вопрос пользователя: {user_input}

Ответь на вопрос, используя предоставленный контекст. Если контекст недостаточен, используй свои знания. Будь кратким и точным."""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history + [
                    {"role": "system", "content": "Ты полезный ассистент. Отвечай точно и по делу."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response['message']['content']
            
            # Сохранение истории
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Ограничение истории
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            return answer, context
        except Exception as e:
            return f"Ошибка: {str(e)}", ""
        
    def save_knowledge_base(self, filename: str = "knowledge_base.json"):
        """Сохранить базу знаний в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.vector_db, f, ensure_ascii=False, indent=2)
    
    def load_knowledge_base(self, filename: str = "knowledge_base.json"):
        """Загрузить базу знаний из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.vector_db = json.load(f)
        except FileNotFoundError:
            print("Файл базы знаний не найден, создается новая")
    
    def save_with_pickle(self, filename: str = "knowledge_base.pkl"):
        """Сохранить с использованием pickle (для эмбеддингов)"""
        with open(filename, 'wb') as f:
            pickle.dump(self.vector_db, f)
    
    def load_with_pickle(self, filename: str = "knowledge_base.pkl"):
        """Загрузить с использованием pickle"""
        try:
            with open(filename, 'rb') as f:
                self.vector_db = pickle.load(f)
        except FileNotFoundError:
            print("Файл базы знаний не найден")

class VoiceProcessor:
    def __init__(self):
        self.tts = VitsTTS()
    
    def speech_to_text(self, audio_file: str) -> str:
        """Преобразование аудио в текст"""
        try:
            recognizer = sr.Recognizer()
            
            # Конвертация в WAV если нужно
            if audio_file.endswith('.mp3'):
                audio = AudioSegment.from_mp3(audio_file)
                wav_file = audio_file.replace('.mp3', '.wav')
                audio.export(wav_file, format='wav')
                audio_file = wav_file
            
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language='ru-RU')
                return text
        except sr.UnknownValueError:
            return "Речь не распознана"
        except Exception as e:
            return f"Ошибка распознавания: {str(e)}"
    
    def text_to_speech(self, text: str, speed: float = 1.0) -> str:
        """Преобразование текста в аудио с помощью VITS"""
        return self.tts.text_to_speech(text, speed)



# Получение списка доступных моделей Ollama
try:
    models = ollama.list()
    available_models = [model['model'] for model in models['models']]
    print("Доступные модели:", available_models)
    default_model = available_models[0] if available_models else "llama2"
except Exception as e:
    print("Ошибка получения списка моделей:", e)
    available_models = ["llama2"]
    default_model = "llama2"

# Инициализация систем
rag_system = RAGChatSystem(model_name=default_model)
voice_processor = VoiceProcessor()

rag_system.load_knowledge_base("knowledge_base.json")

# Добавление примеров (если база пустая)
if not rag_system.vector_db:
    # Добавление примеров документов в базу знаний
    example_documents = [
        "Ollama - это платформа для запуска больших языковых модеей локально. Она позволяет использовать модели как Llama, Mistral и другие без интернета.",
        "Gradio - это библиотека Python для создания веб-интерфейсов для машинного обучения. Проста в использовании и настройке.",
        "RAG (Retrieval-Augmented Generation) - это техника, сочетающая поиск информации и генерацию текста. Улучшает точность ответов модели.",
        "VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) - современная модель для синтеза речи.",
        "Для голосового ввода используется библиотека SpeechRecognition, которая поддерживает различные движки распознавания."
    ]
    rag_system.add_to_knowledge_base(example_documents)
    rag_system.save_knowledge_base()  # Сохранить после добавления

def chat_interface(message: str, audio_input=None, use_voice_input: bool = False, speech_speed: float = 1.0):
    """Основной интерфейс чата"""
    
    # Обработка голосового ввода
    if use_voice_input and audio_input is not None:
        user_input = voice_processor.speech_to_text(audio_input)
    else:
        user_input = message
    
    if not user_input.strip():
        return "Введите сообщение или загрузите аудио", "", None, ""
    
    # Генерация ответа
    response, context = rag_system.generate_response(user_input)
    
    # Генерация аудио ответа
    audio_output = voice_processor.text_to_speech(response, speech_speed)
    
    # Форматирование вывода
    formatted_output = f"""**Пользователь:** {user_input}

**Ответ:** {response}

---
*Контекст RAG:*
{context if context else 'Использованы общие знания модели'}"""
    
    return formatted_output, response, audio_output if audio_output else None, user_input if use_voice_input else ""

def clear_chat():
    """Очистка истории чата"""
    rag_system.conversation_history.clear()
    return "", "", None, ""

def add_text_to_knowledge(text):
    """Добавить текст из ответа в базу знаний"""
    if text and text.strip():
        rag_system.add_to_knowledge_base([text.strip()])
        rag_system.save_knowledge_base()
        return f"Текст добавлен в базу знаний! Размер базы: {len(rag_system.vector_db)} документов"
    return "Текст пустой, нечего добавлять"

# Создание Gradio интерфейса
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Chat с VITS TTS") as demo:
    gr.Markdown("# 🎯 RAG Чат с VITS TTS")
    gr.Markdown("Общайтесь с моделью через текст или голос с использованием RAG и качественным синтезом речи")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Настройки")
            model_selector = gr.Dropdown(
                choices=available_models,
                value=default_model,
                label="Выберите модель Ollama"
            )
            
            voice_input_checkbox = gr.Checkbox(
                label="Использовать голосовой ввод",
                value=False
            )
            
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Голосовой ввод",
                visible=False
            )
            
            speech_speed = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Скорость речи"
            )
            
            add_knowledge_btn = gr.UploadButton(
                "📁 Добавить документы в базу знаний",
                file_types=[".txt"],
                file_count="multiple"
            )
            
            clear_btn = gr.Button("🧹 Очистить историю", variant="secondary")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Диалог",
                height=400,
                show_copy_button=True,
                avatar_images=(
                    "https://media.istockphoto.com/id/1334436084/vector/customer-service-icon-vector.jpg?s=612x612&w=0&k=20&c=K7VcXk-5nOZ25Y6Z8bqQ2pYVnRfqKfR6xQ9Xf6y1jqk=",
                    "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
                )
            )
            
            recognized_text = gr.Textbox(
                label="Распознанный текст",
                interactive=True,  # Сделано редактируемым
                visible=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Текстовый ввод",
                    placeholder="Введите ваше сообщение...",
                    lines=2,
                    scale=4,
                    interactive=True  # Сделано редактируемым
                )
                send_btn = gr.Button("📤 Отправить", variant="primary", scale=1)
            
            gr.Markdown("### Ответ модели")
            with gr.Row():
                text_output = gr.Textbox(
                    label="Текстовый ответ",
                    interactive=True,  # Сделано редактируемым
                    lines=3,
                    scale=4
                )
                add_to_kb_btn = gr.Button("💾 Добавить в базу знаний", scale=1)
            
            add_status = gr.Textbox(
                label="Статус",
                interactive=False,
                visible=False
            )
            
            audio_output = gr.Audio(
                label="Голосовой ответ (VITS)",
                interactive=False,
                type="filepath",
                autoplay=True
            )
    
    # Обработчики событий
    def toggle_voice_input(use_voice: bool):
        return (
            gr.Audio(visible=use_voice), 
            gr.Textbox(visible=not use_voice),
            gr.Textbox(visible=use_voice)
        )
    
    voice_input_checkbox.change(
        toggle_voice_input,
        inputs=[voice_input_checkbox],
        outputs=[audio_input, msg, recognized_text]
    )
    
    # Основная логика отправки сообщения
    def send_message(message, audio, use_voice, speed):
        if not message and not audio:
            return "", "", "", None, "", ""
        
        # Обработка голосового ввода
        if use_voice and audio:
            recognized = voice_processor.speech_to_text(audio)
            if recognized.startswith("Ошибка") or recognized == "Речь не распознана":
                return "", "", recognized, None, recognized, ""
            input_text = recognized
        else:
            input_text = message
        
        if not input_text.strip():
            return "", "", "Текст пустой", None, "", ""
        
        formatted_output, text_response, audio_response, recognized_display = chat_interface(
            input_text, audio, use_voice, speed
        )
        
        # Обновление чатбота
        chat_history = [(input_text, text_response)]
        
        return "", chat_history, recognized_display, text_response, audio_response, ""
    
    send_btn.click(
        send_message,
        inputs=[msg, audio_input, voice_input_checkbox, speech_speed],
        outputs=[msg, chatbot, recognized_text, text_output, audio_output, add_status]
    )
    
    msg.submit(
        send_message,
        inputs=[msg, audio_input, voice_input_checkbox, speech_speed],
        outputs=[msg, chatbot, recognized_text, text_output, audio_output, add_status]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, text_output, audio_output, recognized_text, add_status]
    )
    
    # Обработчик добавления текста в базу знаний
    add_to_kb_btn.click(
        add_text_to_knowledge,
        inputs=[text_output],
        outputs=[add_status]
    )
    
    def add_documents(files):
        if files:
            contents = []
            for file in files:
                try:
                    with open(file.name, 'r', encoding='utf-8') as f:
                        content = f.read()
                        contents.append(content)
                except Exception as e:
                    return f"Ошибка загрузки файла {file.name}: {str(e)}"
            
            rag_system.add_to_knowledge_base(contents)
            rag_system.save_knowledge_base()
            return f"Добавлено {len(files)} документов в базу знаний. Всего документов: {len(rag_system.vector_db)}"
        return "Файлы не выбраны"
    
    add_knowledge_btn.upload(
        add_documents,
        inputs=[add_knowledge_btn],
        outputs=[add_status]
    )
    
    def update_model(model_name):
        rag_system.model_name = model_name
        rag_system.conversation_history.clear()
        return f"Модель изменена на: {model_name}"
    
    model_selector.change(
        update_model,
        inputs=[model_selector],
        outputs=[add_status]
    )

if __name__ == "__main__":    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )