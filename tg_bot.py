import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Включим логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен вашего бота (замените на свой)
TOKEN = 

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_text = f"""
👋 Привет, {user.first_name}!

Я бот для обработки изображений.

📤 Отправь мне картинку, и я:
• Сохраню её
• Покажу информацию о ней
• Могу выполнить базовые операции

Просто отправь мне изображение как фото или файл!
    """
    await update.message.reply_text(welcome_text)

# Обработчик получения изображений
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Получаем информацию о пользователе
        user = update.effective_user
        
        # Отправляем сообщение о начале обработки
        await update.message.reply_text("📥 Получил изображение! Обрабатываю...")
        
        # Получаем фото в максимальном качестве
        photo_file = await update.message.photo[-1].get_file()
        
        # Генерируем имя файла
        file_name = f"photo_{user.id}_{update.message.message_id}.jpg"
        
        # Скачиваем файл
        await photo_file.download_to_drive(file_name)
        
        # Отправляем информацию о файле
        info_text = f"""
✅ Изображение сохранено!

📊 Информация:
• ID пользователя: {user.id}
• Имя: {user.first_name}
• Файл: {file_name}
• Размер файла: {photo_file.file_size // 1024} КБ
        """
        
        await update.message.reply_text(info_text)
        
        # Отправляем обратно уменьшенное изображение
        await update.message.reply_photo(
            photo=photo_file.file_id,
            caption="Вот ваше изображение 📷"
        )
        
    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        await update.message.reply_text("❌ Произошла ошибка при обработке изображения")

# Обработчик получения документов (изображений как файлов)
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    
    # Проверяем, что это изображение
    if document.mime_type and document.mime_type.startswith('image/'):
        await update.message.reply_text("📄 Получил изображение как файл! Обрабатываю...")
        
        # Скачиваем файл
        file_name = f"doc_{update.effective_user.id}_{document.file_name}"
        file = await document.get_file()
        await file.download_to_drive(file_name)
        
        await update.message.reply_text(
            f"✅ Изображение сохранено как {file_name}\n"
            f"📏 Размер: {document.file_size // 1024} КБ\n"
            f"📝 MIME тип: {document.mime_type}"
        )
    else:
        await update.message.reply_text("❌ Пожалуйста, отправьте изображение")

# Обработчик ошибок
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")
    if update and update.message:
        await update.message.reply_text("❌ Произошла ошибка. Попробуйте еще раз.")

# Основная функция
def main():
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    
    # Регистрируем обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    print("🤖 Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()