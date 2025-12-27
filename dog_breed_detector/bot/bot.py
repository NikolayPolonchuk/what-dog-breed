import os
import torch
import asyncio
import logging
from pathlib import Path
from io import BytesIO
from typing import Tuple, List

import hydra
from PIL import Image
import torchvision.transforms as transforms
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Добавляем путь для импортов
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dog_breed_detector.model.vit_model import PretrainViT
from dog_breed_detector.dataset.dataset import DogDataset

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class DogBreedClassifier:
    """Классификатор пород собак"""
    
    def __init__(self, cfg_path: str = "../configs/config.yaml"):
        """Инициализация классификатора"""
        # Загружаем конфиг
        with hydra.initialize(version_base=None, config_path="../configs"):
            self.cfg = hydra.compose(config_name="config")
        
        # Параметры из конфига
        self.image_size = self.cfg.dataset.preprocessing.image_size
        self.resize = self.cfg.dataset.preprocessing.resize
        self.channel_mean = torch.Tensor(self.cfg.dataset.preprocessing.channel_mean)
        self.channel_std = torch.Tensor(self.cfg.dataset.preprocessing.channel_std)
        
        # Создаем трансформации
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.channel_mean, std=self.channel_std),
        ])
        
        # Загружаем модель
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Загружаем метки классов
        self.label_idx2name = self._load_labels()
        
        logger.info(f"Классификатор инициализирован на устройстве: {self.device}")
        logger.info(f"Количество классов: {len(self.label_idx2name)}")
    
    def _load_model(self) -> PretrainViT:
        """Загрузка обученной модели"""
        model_path = Path("model/model.pth")
        
        if not model_path.exists():
            # Пытаемся найти модель в других местах
            possible_paths = [
                Path("../model/model.pth"),
                Path("dog_breed_detector/model/model.pth"),
                Path(__file__).parent.parent / "model" / "model.pth"
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
        
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден. Искали в: {model_path}")
        
        logger.info(f"Загружаем модель из: {model_path}")
        
        # Создаем модель
        model = PretrainViT(self.cfg)
        
        # Загружаем веса
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_labels(self) -> List[str]:
        """Загрузка меток классов из датасета"""
        try:
            # Создаем датасет для получения меток
            dataset = DogDataset(
                img_path=f"{self.cfg.dataset.paths.data_dir}/{self.cfg.dataset.paths.train_images}",
                csv_path=f"{self.cfg.dataset.paths.data_dir}/{self.cfg.dataset.paths.train_labels}",
                transform=None
            )
            return dataset.label_idx2name
        except Exception as e:
            logger.error(f"Ошибка загрузки меток: {e}")
            # Возвращаем заглушку с номерами классов
            return [f"Class_{i}" for i in range(self.cfg.model.model.num_classes)]
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Предсказание породы собаки по изображению"""
        try:
            # Преобразуем изображение
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence_value = confidence.item()
                predicted_class = self.label_idx2name[predicted_idx.item()]
            
            return predicted_class, confidence_value
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise


class DogBreedBot:
    """Телеграм бот для определения породы собак"""
    
    def __init__(self, token: str, classifier: DogBreedClassifier):
        self.token = token
        self.classifier = classifier
        self.application = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        welcome_text = (
            f"Привет, {user.first_name}! 👋\n\n"
            "Я бот для определения породы собак 🐶\n\n"
            "Просто отправь мне фото собаки, и я определю её породу!\n\n"
            "Команды:\n"
            "/start - Начало работы\n"
            "/help - Помощь\n"
            "/about - О боте"
        )
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = (
            "📋 **Помощь по использованию бота:**\n\n"
            "1. Отправьте фото собаки (в формате JPG/PNG)\n"
            "2. Я проанализирую изображение и определю породу\n"
            "3. Покажу результат с вероятностью правильности\n\n"
            "⚠️ **Важно:**\n"
            "- Изображение должно быть четким\n"
            "- Собака должна быть хорошо видна\n"
            "- Работает только с фотографиями собак\n\n"
            "Команды:\n"
            "/start - Начало работы\n"
            "/help - Помощь\n"
            "/about - О боте"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /about"""
        about_text = (
            "🤖 **О боте:**\n\n"
            "Этот бот использует нейронную сеть Vision Transformer (ViT-L/16),\n"
            "обученную на датасете Stanford Dogs.\n\n"
            "📊 **Технические детали:**\n"
            "- Модель: Vision Transformer L/16\n"
            "- Количество классов: 120 пород собак\n"
            "- Точность на валидации: ~85%\n\n"
            "👨‍💻 **Разработчик:**\n"
            "Создан как часть проекта классификации пород собак"
        )
        await update.message.reply_text(about_text, parse_mode='Markdown')
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик фотографий"""
        try:
            # Отправляем сообщение о начале обработки
            processing_msg = await update.message.reply_text(
                "🔍 Анализирую изображение... Пожалуйста, подождите!"
            )
            
            # Получаем фото максимального качества
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            
            # Конвертируем в PIL Image
            image = Image.open(BytesIO(photo_bytes)).convert('RGB')
            
            # Предсказываем породу
            breed, confidence = self.classifier.predict(image)
            
            # Форматируем результат
            confidence_percent = confidence * 100
            
            # Определяем эмодзи в зависимости от уверенности
            if confidence_percent > 80:
                confidence_emoji = "🎯"
            elif confidence_percent > 60:
                confidence_emoji = "✅"
            elif confidence_percent > 40:
                confidence_emoji = "🤔"
            else:
                confidence_emoji = "❓"
            
            # Формируем ответ
            result_text = (
                f"{confidence_emoji} **Результат:**\n\n"
                f"🐕 **Порода:** {breed}\n"
                f"📊 **Уверенность:** {confidence_percent:.1f}%\n\n"
            )
            
            # Добавляем рекомендации
            if confidence_percent > 70:
                result_text += "✅ Высокая уверенность в определении!"
            elif confidence_percent > 40:
                result_text += "⚠️ Средняя уверенность. Попробуйте отправить более четкое фото."
            else:
                result_text += "❓ Низкая уверенность. Возможно:\n- Это не собака\n- Слишком маленькое изображение\n- Нестандартный ракурс"
            
            # Обновляем сообщение с результатом
            await processing_msg.edit_text(result_text, parse_mode='Markdown')
            
            logger.info(f"Обработано фото от пользователя {update.effective_user.id}: {breed} ({confidence_percent:.1f}%)")
            
        except Exception as e:
            logger.error(f"Ошибка обработки фото: {e}")
            error_text = (
                "❌ Произошла ошибка при обработке изображения.\n\n"
                "Пожалуйста, попробуйте:\n"
                "1. Отправить другое фото\n"
                "2. Убедиться, что на фото собака\n"
                "3. Проверить качество изображения"
            )
            await update.message.reply_text(error_text)
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик документов (изображений, отправленных как файл)"""
        try:
            document = update.message.document
            
            # Проверяем, что это изображение
            if document.mime_type and document.mime_type.startswith('image/'):
                # Отправляем сообщение о начале обработки
                processing_msg = await update.message.reply_text(
                    "🔍 Анализирую изображение... Пожалуйста, подождите!"
                )
                
                # Скачиваем файл
                file = await document.get_file()
                photo_bytes = await file.download_as_bytearray()
                
                # Конвертируем в PIL Image
                image = Image.open(BytesIO(photo_bytes)).convert('RGB')
                
                # Предсказываем породу
                breed, confidence = self.classifier.predict(image)
                confidence_percent = confidence * 100
                
                # Формируем ответ
                result_text = (
                    f"✅ **Результат:**\n\n"
                    f"🐕 **Порода:** {breed}\n"
                    f"📊 **Уверенность:** {confidence_percent:.1f}%"
                )
                
                await processing_msg.edit_text(result_text, parse_mode='Markdown')
            else:
                await update.message.reply_text(
                    "📄 Пожалуйста, отправьте изображение собаки, а не документ!"
                )
                
        except Exception as e:
            logger.error(f"Ошибка обработки документа: {e}")
            await update.message.reply_text("❌ Ошибка обработки файла!")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        text = update.message.text.lower()
        
        if any(word in text for word in ['привет', 'hello', 'hi']):
            await update.message.reply_text(
                "Привет! Отправь мне фото собаки, и я определю её породу! 🐶"
            )
        elif any(word in text for word in ['спасибо', 'thanks', 'thank']):
            await update.message.reply_text(
                "Всегда рад помочь! 🐕"
            )
        else:
            await update.message.reply_text(
                "Я понимаю только фотографии собак! 📸\n"
                "Отправь мне фото, и я определю породу."
            )
    
    def setup_handlers(self):
        """Настройка обработчиков"""
        # Команды
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("about", self.about))
        
        # Фотографии
        self.application.add_handler(MessageHandler(
            filters.PHOTO, self.handle_photo
        ))
        
        # Документы (изображения как файлы)
        self.application.add_handler(MessageHandler(
            filters.Document.IMAGE, self.handle_document
        ))
        
        # Текстовые сообщения
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self.handle_text
        ))
    
    async def run(self):
        """Запуск бота"""
        # Создаем Application
        self.application = Application.builder().token(self.token).build()
        
        # Настраиваем обработчики
        self.setup_handlers()
        
        # Запускаем бота
        logger.info("Бот запускается...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        # Ждем остановки
        logger.info("Бот запущен и готов к работе!")
        await asyncio.Event().wait()
    
    def run_polling(self):
        """Запуск бота в режиме polling (блокирующий)"""
        # Создаем Application
        self.application = Application.builder().token(self.token).build()
        
        # Настраиваем обработчики
        self.setup_handlers()
        
        # Запускаем polling
        logger.info("Бот запускается в режиме polling...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Основная функция запуска бота"""
    # Токен бота (замените на свой или установите переменную окружения)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        # Попробуем прочитать из файла
        token_file = Path("bot_token.txt")
        if token_file.exists():
            with open(token_file, 'r') as f:
                token = f.read().strip()
        else:
            logger.error("Токен бота не найден!")
            logger.info("Установите переменную окружения TELEGRAM_BOT_TOKEN или создайте файл bot_token.txt")
            return
    
    try:
        # Инициализируем классификатор
        logger.info("Инициализация классификатора...")
        classifier = DogBreedClassifier()
        
        # Создаем и запускаем бота
        bot = DogBreedBot(token, classifier)
        bot.run_polling()
        
    except Exception as e:
        logger.error(f"Ошибка запуска бота: {e}")
        raise


if __name__ == "__main__":
    main()