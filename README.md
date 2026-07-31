<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-FFD21E.svg" alt="Hugging Face">
    <a href="https://www.kaggle.com/datasets/antonbezzaborov/docvqa-ru-eng-v1">
        <img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg?logo=Kaggle&logoColor=white" alt="Kaggle Dataset">
    </a>
</p>

# Clever Document Assistant (Cyrillic Document VQA)

**Умный ИИ-помощник для работы с документами на русском языке (OCR, VQA, Layout Analysis)**

Репозиторий представляет собой фреймворк для построения систем понимания документов (Dociment AI) с использованием современных визуально-лингвистических моделей (VLM). Проект включает полный пайплайн: от консолидации данных и генерации синтетики до дообучения моделей (SFT, GRPO) и деплоя в виде Telegram-бота.

---

## Основная идея

Проект ориентирован на исследовательскую работу и быстрое прототипирование (notebooks + reusable code).
Clever Document Assistant — это набор инструментов для:

1. **Layout Analysis & Feature Extraction:** Понимание структуры документа и извлечение визуальных признаков.
2. **Visual Question Answering (VQA):** Точные ответы на вопросы пользователя по содержимому отсканированного документа или PDF-файла.
3. **Inference & UI:** Запуск моделей локально или через Telegram-бота для быстрой демонстрации возможностей.

---

## Ключевые особенности

- **SOTA Архитектуры:** Использование передовых генеративных vision-моделей **Florence-2** (Large) и **Qwen2.5-VL** (32B Instruct).
- **Продвинутое дообучение (Fine-Tuning):** Реализованы воспроизводимые пайплайны обучения с использованием **QLoRA**, Supervised Fine-Tuning (**SFT**) и Group Relative Policy Optimization (**GRPO**).
- **Интеграция Depth-Breadth Fusion:** Использование инновационных архитектурных подходов (вдохновленных *Florence-VL*) для слияния визуальных признаков с разных слоев (depth) и под разные промпты (breadth) перед подачей в LLM.
- **Собственный публичный датасет:** Собран, очищен и опубликован на Kaggle объединенный русскоязычный мультимодальный датасет для задачи Document VQA.

---

## Консолидация данных

Ввиду острой нехватки качественных русскоязычных данных для задач Document AI и VQA, для обучения и оценки моделей были собраны и объединены 5 ключевых наборов данных — практически все релевантные open-source источники, доступные на данный момент:

1. **[ruCLEVR](https://huggingface.co/datasets/MERA-evaluation/ruCLEVR)** — русскоязычная версия набора для визуального рассуждения (MERA-evaluation).  
2. **[ruVQA](https://huggingface.co/datasets/MERA-evaluation/ruVQA)** — русскоязычный dataset для задачи Visual Question Answering (MERA-evaluation).  
3. **[MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru)** — русскоязычная версия мультимодального бенчмарка MMBench (deepvk).  
4. **[MWS-Vision-Bench](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench)** — dataset для понимания технических диаграмм (MTSAIR).  
5. **[Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)** — набор данных для понимания документов (HuggingFaceM4, используются только документы с одним изображением). 

**Итоговый объединённый набор данных опубликован в открытом доступе на Kaggle:**
[DocVQA-RU-ENG-v1](https://www.kaggle.com/datasets/antonbezzaborov/docvqa-ru-eng-v1)

> *Примечание: Объединение и подготовка данных выполняется в ноутбуке `1.0-data-consolidation.ipynb`. Полностью поддерживается схема HuggingFace Datasets.*

---

## Навигация по репозиторию

### Модели (`models/`)
В каталоге `models/` присутствуют две основные директории:
* `pre_trained/` — образы/артефакты предобученных моделей (например, `florence_2_large`, `qwen2_5_vl_32B_Instruct`).
* `fine_tuned/` — веса адаптеров (LoRA) и конфигурационные файлы для дообученных версий архитектур (`training_args`, `adapter_config` и др.).

> **Важно:** Если вы планируете воспроизводить эксперименты с дообучением Qwen-32B, убедитесь, что у вас установлены подходящие версии transformers и accelerate. Поскольку Qwen2.5-VL — новая архитектура, для её корректной загрузки и работы с QLoRA требуется свежая кодовая база Hugging Face. Также убедитесь, что у вас имеется достаточный объем VRAM.

### Ноутбуки (`notebooks/`)

Директория содержит рабочие блокноты, логически разделённые по архитектурам и этапам:
* `qwen2_5_vl_32B_Instruct/` — data processing, training (QLoRA, SFT/GRPO), inference/evaluation для VQA и LLM.
* `florence_2_large/` — data synthesis, fine-tuning и inference/evaluation (в т.ч. подсчет метрик WER/CER и LLM-as-a-judge).
* `florence_vl/` — feature extraction и вспомогательные скрипты для экспериментов с Depth-Breadth Fusion.

---

## Структура проекта 
```
clever-document-assistant-ru/
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── setup.cfg
├── pyproject.toml
│
├── docs/
│   └── index.md
│
├── models/
│   ├── pre_trained/
│   │   ├── florence_2_large/
│   │   │   ├── config.json
│   │   │   ├── preprocessor_config.json
│   │   │   ├── ...
│   │   │   ├── tokenizer_config.json
│   │   │   └── README.md
│   │   └── qwen2_5_vl_32B_Instruct/
│   │       ├── config.json
│   │       ├── tokenizer_config.json
│   │       ├── ...
│   │       ├── special_tokens_map.json
│   │       └── README.md
│   └── fine_tuned/
│       ├── florence_2_large/
│       │   ├── adapter_config.json
│       │   ├── ...
│       │   ├── training_args.json
│       │   └── README.md
│       └── qwen2_5_vl_32B_Instruct/
│           ├── adapter_config.json
│           ├── ...
│           ├── training_args.json
│           └── README.md
│
├── notebooks/
│   ├── qwen2_5_vl_32B_Instruct/
│   │   ├── data_processing/
│   │   │   └── 1.0-data-consolidation.ipynb
│   │   ├── training/
│   │   │   ├── 2.0-qwen-qlora-sft.ipynb
│   │   │   └── 2.1-qwen-qlora-grpo.ipynb
│   │   └── inference/
│   │       ├── 3.0-qwen-evaluation-vqa.ipynb
│   │       └── 3.1-qwen-evaluation-llm.ipynb
│   ├── florence_2_large/
│   │   ├── data_processing/
│   │   │   └── 4.0-data-syntesis.ipynb
│   │   ├── training/
│   │   │   └── 5.0-florence-finetuning.ipynb
│   │   └── inference/
│   │       ├── 6.0-florence-evaluation-test.ipynb
│   │       └── 6.1-florence-evaluation-wer-cer.ipynb
│   └── florence_vl/
│       └──  data_processing/
│           └── 7.0-feature-extraction.ipynb
│
├── references/
│   ├── papers/
│   │   ├── document_ai_comparative_study_layout_analysis.pdf
│   │   ├── enhancing_document_understanding_contrastive_learning.pdf
│   │   ├── florence_2_unified_vision_tasks.pdf
│   │   ├── florence_vl_depth_breadth_fusion.pdf
│   │   └── layoutlmv3_pretraining_document_ai.pdf
│   └── datasets/
│
├── reports/
│   └── figures/
│
└── clever_document_assistant_ru/
    ├── __init__.py
    └── bot/
        ├── telegram_bot.py
        └── inference_model.py
```

---

## Опора на исследования (`references/`)
Архитектурные решения в данном проекте опираются на следующие научные работы:
1. [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://github.com/antonbezzaborov/clever-document-assistant/blob/main/references/papers/florence_2_unified_vision_tasks.pdf) (Microsoft Azure AI).
2. [Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](https://github.com/antonbezzaborov/clever-document-assistant/blob/main/references/papers/florence_vl_depth_breadth_fusion.pdf) (UMD & Microsoft Research).
3. [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://github.com/antonbezzaborov/clever-document-assistant/blob/main/references/papers/layoutlmv3_pretraining_document_ai.pdf) (Microsoft Research).
4. [Document AI: A Comparative Study of Transformer-Based, Graph-Based Models, and Convolutional Neural Networks For Document Layout Analysis](https://github.com/antonbezzaborov/clever-document-assistant/blob/main/references/papers/document_ai_comparative_study_layout_analysis.pdf).
5. [Enhancing Visual Document Understanding with Contrastive Learning in Large Visual-Language Models](https://github.com/antonbezzaborov/clever-document-assistant/blob/main/references/papers/enhancing_document_understanding_contrastive_learning.pdf).

---
