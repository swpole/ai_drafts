import torch
from scipy.io import wavfile
from transformers import VitsModel, AutoTokenizer
from pathlib import Path
from typing import Optional, Union
import logging

class TextToSpeech:
    def __init__(self, 
                 model_name: str = "facebook/mms-tts-rus",
                 device: Optional[str] = None):
        """
        Инициализация модели преобразования текста в речь
        
        Args:
            model_name: название предобученной модели
            device: устройство для вычислений ('cuda', 'cpu', или None для автоопределения)
        """
        self.model_name = model_name
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def _setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            self.logger.info(f"Загрузка модели {self.model_name} на устройство {self.device}")
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.is_loaded = True
            self.logger.info("Модель успешно загружена")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def generate_speech(self, 
                       text: str,
                       output_path: Optional[str] = None,
                       sampling_rate: Optional[int] = None,
                       normalize_audio: bool = True) -> torch.Tensor:
        """
        Генерация речи из текста
        
        Args:
            text: текст для преобразования
            output_path: путь для сохранения аудиофайла
            sampling_rate: частота дискретизации (если None - используется из модели)
            normalize_audio: нормализовать ли аудио для WAV формата
            
        Returns:
            Тензор с аудиоданными
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            self.logger.info(f"Генерация речи для текста: {text[:50]}...")
            
            # Токенизация текста
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Генерация аудио
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            # Сохранение в файл, если указан путь
            if output_path:
                self.save_audio(output, output_path, sampling_rate, normalize_audio)
            
            return output.cpu()
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации речи: {e}")
            raise
    
    def save_audio(self, 
                  audio_tensor: torch.Tensor,
                  output_path: str,
                  sampling_rate: Optional[int] = None,
                  normalize_audio: bool = True):
        """
        Сохранение аудио тензора в файл
        
        Args:
            audio_tensor: тензор с аудиоданными
            output_path: путь для сохранения
            sampling_rate: частота дискретизации
            normalize_audio: нормализовать ли аудио
        """
        try:
            # Преобразование в numpy массив
            audio = audio_tensor.cpu().numpy()
            audio = audio.squeeze()
            
            # Нормализация для WAV формата
            if normalize_audio:
                audio = (audio * 32767).astype('int16')
            
            # Определение частоты дискретизации
            sr = sampling_rate if sampling_rate else self.model.config.sampling_rate
            
            # Создание директории, если не существует
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Сохранение файла
            wavfile.write(output_path, sr, audio)
            self.logger.info(f"Аудио сохранено в: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении аудио: {e}")
            raise
    
    def batch_generate(self, 
                      texts: list,
                      output_dir: str = "output",
                      output_prefix: str = "speech",
                      sampling_rate: Optional[int] = None):
        """
        Пакетная генерация речи для нескольких текстов
        
        Args:
            texts: список текстов для преобразования
            output_dir: директория для сохранения файлов
            output_prefix: префикс для имен файлов
            sampling_rate: частота дискретизации
        """
        results = []
        
        # Создание директории
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, text in enumerate(texts):
            try:
                output_path = f"{output_dir}/{output_prefix}_{i+1:03d}.wav"
                audio = self.generate_speech(text, output_path, sampling_rate)
                results.append({
                    'text': text,
                    'audio': audio,
                    'file_path': output_path
                })
                self.logger.info(f"Обработан текст {i+1}/{len(texts)}")
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке текста {i+1}: {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'file_path': None
                })
        
        return results

# Пример использования
if __name__ == "__main__":
    # Создание экземпляра класса
    tts = TextToSpeech()
    
    # Одиночная генерация
    audio = tts.generate_speech(
        text="Привет, это тест преобразования текста в речь",
        output_path="output/hello.wav"
    )
    
    # Пакетная генерация
    texts = [
        "Первое предложение для теста",
        "Второе предложение для проверки работы",
        "Третье предложение для демонстрации возможностей"
    ]
    
    results = tts.batch_generate(
        texts=texts,
        output_dir="output/batch",
        output_prefix="sample"
    )
    
    print("Генерация завершена!")