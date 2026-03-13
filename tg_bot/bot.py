import asyncio
import io
import aiohttp
import base64
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message, ErrorEvent
from aiogram.enums import ContentType, ParseMode
from aiogram.filters import Command
import logging
import traceback
from PIL import Image
from inference_model import generate_answer
import fitz

# Настройки
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

# Инициализация
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Хранилище пользовательских данных в памяти
user_data = {}
user_size_data = {}

# Логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Обработчик команды /start
@dp.message(Command("start"))
async def start_handler(message: Message):
    try:
        logger.info(f"🚀 /start от пользователя {message.from_user.id}")
        welcome_text = """
🤖 <b>Здравствуйте. Я ваш умный помощник для обработки документов!</b>

Я могу:
• 📷 Обрабатывать изображения через OCR
• 📄 Извлекать текст из PDF, JPG и PNG файлов
• 🧠 Анализировать контент с помощью AI

<b>Как работать:</b>
1. Отправьте мне изображения или PDF файлы
2. Я извлеку из них текст
3. Отправьте вопрос по документу
4. Получите ответ!

Просто отправьте файл чтобы начать!
        """
        await message.answer(welcome_text)
        logger.info(f"✅ /start успешно обработан для {message.from_user.id}")
    except Exception as e:
        logger.error(f"❌ Ошибка в /start: {e}\n{traceback.format_exc()}")
        await message.answer("Произошла ошибка при обработке команды /start")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    try:
        logger.info(f"🆘 /help от пользователя {message.from_user.id}")
        help_text = """
<b>Инструкция по использованию:</b>

1. <b>Отправьте файлы</b> - изображение (jpg, png) или PDF
2. <b>Дождитесь обработки</b> - я извелку текст через OCR
3. <b>Отправьте текст с вопросом</b> - напишите что вас интересует в документе
4. <b>Получите ответ</b> - модель проанализирует содержимое

<b>Поддерживаемые форматы:</b>
• Изображения: JPG, PNG, JPEG, BMP, TIFF
• Документы: PDF

<b>После обработки файла вы не сможете добавить новые, пока не используете /restart</b>
        """
        await message.answer(help_text)
        logger.info(f"✅ /help успешно обработан для {message.from_user.id}")
    except Exception as e:
        logger.error(f"❌ Ошибка в /help: {e}\n{traceback.format_exc()}")
        await message.answer("Произошла ошибка при обработке команды /help")


# Обработчик команды /restart
@dp.message(Command("restart"))
async def clear_handler(message: Message):
    user_id = message.from_user.id
    logger.info(f"🔄 /restart от пользователя {user_id}")
    try:
        if user_id in user_data:
            del user_data[user_id]
            logger.info(f"✅ Данные очищены для {user_id}")
            await message.answer("✅ Данные успешно очищены. Теперь вы можете отправить новый файл.")
        else:
            logger.info(f"✅ Нет данных для очистки у {user_id}")
            await message.answer("✅ Нет данных для очистки. Вы можете отправить файл.")
    except Exception as e:
        logger.error(f"❌ Ошибка в /restart: {e}\n{traceback.format_exc()}")
        await message.answer("❌ Произошла ошибка при очистке данных")


# Скачивание файла из Telegram
async def download_file(file_id: str, user_id: str) -> tuple[bytes, str] | tuple[None, None]:
    try:
        logger.debug(f"📥 Скачивание файла {file_id}")
        file = await bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"

        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status != 200:
                    logger.error(f"❌ Ошибка скачивания файла: статус {response.status}")
                    return None, None, None

                file_data = await response.read()
                logger.debug(f"✅ Файл скачан, размер: {len(file_data)} байт")

                # Определяем тип файла
                file_extension = file.file_path.split('.')[-1].lower() if '.' in file.file_path else ''
                if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif']:
                    file_type = "image"
                elif file_extension == 'pdf':
                    file_type = "pdf"
                else:
                    file_type = "document"

                logger.debug(f"📄 Тип файла определен как: {file_type}, размер файла: {file.file_size}")
                # Добавление размера скаченного файла пользователя к суммарному размеру
                if user_id in user_size_data:
                    user_size_data[user_id] += file.file_size
                else:
                    user_size_data[user_id] = file.file_size

                return file_data, file_type

    except Exception as ex:
        logger.error(f"Error downloading file: {ex}\n{traceback.format_exc()}")
        return None, None


# Обработчик только для файлов (без текста)
@dp.message(
    F.content_type.in_({
        ContentType.PHOTO,
        ContentType.DOCUMENT
    })
)
async def handle_files(message: Message):
    user_id = message.from_user.id
    logger.debug(f"📎 Получен файл от {user_id}: тип={message.content_type}")

    try:
        file_data, file_type = None, None

        if message.photo:
            file_id = message.photo[-1].file_id
            file_data, file_type = await download_file(file_id, user_id)
        elif message.document:
            file_id = message.document.file_id
            file_data, file_type = await download_file(file_id, user_id)

        if file_data and file_type:
            # Проверяем поддерживаемые форматы файлов
            if file_type not in ["image", "pdf"]:
                await message.answer(f"❌ Формат файла не поддерживается. Отправьте изображение или PDF.")
                logger.warning(f"⚠️ Неподдерживаемый формат файла от {user_id}: {file_type}")
                return
            if user_id in user_data:
                file = {
                    "type": file_type,
                    "data": base64.b64encode(file_data).decode('utf-8')
                }
                user_data[user_id].append(file)
            else:
                # Сохраняем информацию о файле (только один файл)
                user_data[user_id] = [{
                    "type": file_type,
                    "data": base64.b64encode(file_data).decode('utf-8')
                }]
            await message.answer(
                f"✅ Файл ({file_type}) сохранен. Теперь отправьте текст с вашим вопросом к документу или другие документы.")
            logger.debug(f"💾 Файл сохранен для {user_id}")
        else:
            await message.answer("❌ Не удалось обработать файл")
            logger.error(f"❌ Ошибка обработки файла от {user_id}")

    except Exception as ex:
        logger.error(f"❌ Ошибка в handle_files: {ex}\n{traceback.format_exc()}")
        await message.answer("❌ Произошла ошибка при обработке файла")


# Обработчик текста (кроме команд) - для приема промпта от пользователя
@dp.message(F.content_type == ContentType.TEXT)
async def handle_text(message: Message):
    user_id = message.from_user.id

    # Пропускаем команды - они обрабатываются отдельно
    if message.text.startswith('/'):
        logger.debug(f"⚡ Команда {message.text} передана другому обработчику")
        return
    logger.debug(f"📝 Получен текст от {user_id}: {message.text[:50]}...")

    try:
        # Проверяем наличие файла
        if user_id not in user_data:
            logger.warning(f"⚠️ Пользователь {user_id} не имеет файла")
            await message.answer("❌ Нет файла для запроса. Сначала отправьте файл (изображение или PDF).")
            return

        # Получаем текст вопроса
        question = message.text

        if not question.strip():
            await message.answer("❌ Пожалуйста, укажите вопрос")
            return

        logger.info(f"🚀 Начинаем обработку запроса для {user_id}")
        await process_query(message, user_id, question)

    except Exception as ex:
        logger.error(f"❌ Ошибка в handle_text: {ex}\n{traceback.format_exc()}")
        await message.answer("❌ Произошла ошибка при обработке текста")


# Обработчик неподдерживаемых типов сообщений
@dp.message()
async def handle_unsupported_types(message: Message):
    user_id = message.from_user.id
    content_type = message.content_type

    logger.warning(f"⚠️ Получено неподдерживаемое сообщение от {user_id}: тип={content_type}")

    # Определяем тип контента для понятного сообщения пользователю
    content_type_names = {
        ContentType.TEXT: "текстовые сообщения",
        ContentType.VIDEO: "видео",
        ContentType.VOICE: "голосовые сообщения",
        ContentType.VIDEO_NOTE: "кружочки",
        ContentType.STICKER: "стикеры",
        ContentType.AUDIO: "аудиофайлы",
        ContentType.ANIMATION: "GIF-анимации",
        ContentType.CONTACT: "контакты",
        ContentType.LOCATION: "геолокации",
        ContentType.POLL: "опросы",
        ContentType.DICE: "кости",
    }

    content_name = content_type_names.get(content_type, "этот тип сообщений")

    unsupported_text = f"""❌ Извините, но {content_name} не поддерживаются.
Отправьте изображение или PDF файл, затем отправьте текст с вашим вопросом."""

    await message.answer(unsupported_text)
    logger.debug(f"⚠️ Уведомление о неподдерживаемом формате отправлено {user_id}")


# Обработка запроса к модели
async def process_query(message: Message, user_id: int, question: str):
    try:
        if user_size_data[user_id] >= 1048576:
            await message.answer("❌ Превышен размер документов. Ограничение до 1 Мб")
            await message.answer("Данные очищены")
            del user_size_data[user_id]
            del user_data[user_id]
            return

        logger.debug(f"🔧 process_query начат для {user_id}")
        await message.answer("⏳ Обрабатываю запрос...")
        # Получаем сохраненный файлы
        prepare_data = []
        for file in user_data[user_id]:
            prepare_data.append((base64.b64decode(file['data']), file['type']))

        # Подготавливаем данные для модели
        images, prompt = prepare_data_for_model(prepare_data, question)

        # Получаем ответ от модели
        answer = generate_answer(images, prompt)

        # Отправляем ответ пользователю
        await send_response(message, answer)

        logger.info(f"✅ Запрос успешно обработан для {user_id}")

        # Очистка данных после обработки запроса
        if user_id in user_data:
            del user_data[user_id]

    except Exception as e:
        logger.error(f"❌ Ошибка в process_query: {e}\n{traceback.format_exc()}")
        await message.answer("❌ Произошла ошибка при обработке запроса к модели")


# Функция для подготовки данных к запросу модели
def prepare_data_for_model(files: list[tuple[bytes, str]], question: str) -> tuple[list[Image.Image], str]:
    """
    Подготавливает данные для запроса к модели.

    Args:
        files: набор данных в виде изображений и pdf файлов
        question: Текст вопроса пользователя
    Returns:
        tuple: (PIL.Image объект, текст запроса)
    """
    try:
        logger.debug("🛠️ Подготовка данных для модели")
        images = []
        for file in files:
            if file[1] == "pdf":
                # # Конвертируем PDF в изображение
                logger.debug("📄 Конвертируем PDF в изображение")
                pages = fitz.open(stream=file[0], filetype=file[1])
                if pages:
                    for page in range(len(pages)):
                        pix = pages.load_page(page).get_pixmap(dpi=200)
                        mode = "RGBA" if pix.alpha else "RGB"
                        image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                        images.append(image)
                    logger.debug(f"✅ PDF сконвертирован в изображение")
                else:
                    raise ValueError("Не удалось конвертировать PDF в изображение")
            else:
                # Открываем изображение
                image = Image.open(io.BytesIO(bytes(file[0])))
                images.append(image)
                logger.debug(f"✅ Изображение загружено, размер: {image.size}")

        # Формируем полный запрос
        prompt = f"Вопрос: {question}\n\nПроанализируй содержимое документа и дай развернутый ответ."

        return images, prompt

    except Exception as e:
        logger.error(f"❌ Ошибка при подготовке данных: {e}\n{traceback.format_exc()}")
        raise


# Отправка ответа пользователю
async def send_response(message: Message, response_text: str):
    try:
        logger.debug(f"📤 Отправка ответа: {response_text[:100]}...")
        await message.answer(response_text, parse_mode=None)
        logger.debug("✅ Ответ отправлен как текст")

    except Exception as e:
        logger.error(f"❌ Ошибка в send_response: {e}\n{traceback.format_exc()}")
        await message.answer("❌ Произошла ошибка при отправке ответа")


# Глобальный обработчик ошибок для aiogram 3.x
@dp.error()
async def error_handler(event: ErrorEvent):
    logger.error(f"💥 Глобальная ошибка: {event.exception}\n{traceback.format_exc()}")
    return True


# Функция для проверки работы бота
async def test_bot():
    logger.info("🧪 Запуск тестовой функции")
    try:
        me = await bot.get_me()
        logger.info(f"🤖 Бот @{me.username} успешно инициализирован")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации бота: {e}")
        return False


# Главная функция запуска бота
async def main():
    logger.info("🚀 Бот запускается...")
    # Проверяем работу бота
    if not await test_bot():
        logger.error("❌ Не удалось инициализировать бота")
        return
    logger.info("✅ Бот успешно инициализирован, начинаем пуллинг...")
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка при пуллинге: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}\n{traceback.format_exc()}")
