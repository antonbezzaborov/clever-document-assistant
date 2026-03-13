from unsloth import FastVisionModel
from transformers import TextStreamer
import torch
import os


# === Загрузка модели один раз при старте сервера ===
model_path = os.getenv("MODEL_PATH", "/home/jupyter/datasphere/project/qwen2.5-vl-32b-qlora-a100-copy")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_path,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)
print("Модель успешно загружена и готова к инференсу!")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)


# === Основная функция для одного запроса ===
def generate_answer_one_img(image, question, max_new_tokens=256):
    """
    image: PIL.Image, bytes, или путь к изображению (в зависимости от того, как подаёшь в tokenizer)
    question: str — вопрос пользователя
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]

    # Преобразуем в формат, понятный модели
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Генерация текста
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            min_p=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодируем ответ
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


# === Обновленная функция для обработки нескольких изображений ===
def generate_answer(images: list, question: str, max_new_tokens: int = 256):
    """
    Генерирует ответ на основе списка изображений и текстового вопроса.

    Args:
        images (list): Список изображений. Элементы могут быть PIL.Image,
                       bytes или путями к файлам.
        question (str): Вопрос пользователя.
        max_new_tokens (int): Максимальное количество новых токенов для генерации.
    
    Returns:
        str: Сгенерированный и декодированный ответ модели.
    """
    image_contents = [{"type": "image", "image": img} for img in images]
    
    messages = [
        {"role": "user", "content": image_contents + [
            {"type": "text", "text": question}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(
        images,  # <--- Ключевое изменение
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            min_p=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодируем только сгенерированную часть ответа
    # Находим, где заканчивается промпт и начинается ответ
    prompt_len = inputs['input_ids'].shape[1]
    print(prompt_len)
    decoded_answer = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    return decoded_answer.strip()