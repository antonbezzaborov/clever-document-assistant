<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Clever Document Assistant (RU)

**Умный помощник с документами**

Репозиторий содержит набор средств и экспериментов по построению ассистента для понимания документов с использованием современных визуально-лингвистических моделей (например, Florence 2, Qwen 2.5 VL и др.). Проект включает предобученные и дообученные модели, ноутбуки с пайплайнами обработки данных, тренировками и оценкой, а также телеграм-бот для демонстрации работы.

---

## Основная идея

Clever Document Assistant — это набор инструментов для:

* извлечения признаков и понимания структуры документов (layout),
* ответов на вопросы по изображению/пдф документа (VQA — Visual Question Answering),
* запуска инференса локально или через телеграм-бота для быстрой демонстрации.

Проект ориентирован на исследовательскую работу и прототипирование (notebooks + reusable code).

---

## Модели

В каталоге `models/` присутствуют две основные папки: `pre_trained/` и `fine_tuned/`.

* `pre_trained/` — образы/артефакты предобученных моделей (например, `florence_2_large`, `qwen2_5_vl_32B_Instruct`).
* `fine_tuned/` — адаптеры/веса и конфиги для дообученных версий тех же архитектур (training_args, adapter_config и др.).


> Если вы планируете воспроизводить эксперимент с дообучением: убедитесь, что у вас есть подходящая версия `transformers` и `accelerate`, а также достаточный VRAM для выбранной модели.

---

## Ноутбуки

Папка `notebooks/` содержит рабочие блокноты, разделённые по моделям и этапам:

* `qwen2_5_vl_32B_Instruct/` — data_processing, training (QLoRA, SFT/GRPO ), inference/evaluation для VQA и LLM.
* `florence_2_large/` — data synthesis, fine-tuning и inference/evaluation ноутбуки.
* `florence_vl/` — feature extraction и вспомогательные скрипты.
---

## Датасеты

В проекте используются несколько открытых наборов данных, предназначенных для задач визуального понимания документов и Visual Question Answering (VQA). Они применяются как для обучения, так и для оценки качества моделей.
Датасеты, использованные для обучения модели Qwen2.5-VL

Проект использует пять русскоязычных и мультимодальных датасетов для обучения и дообучения модели Qwen2.5-VL:

1. **[ruCLEVR](https://huggingface.co/datasets/MERA-evaluation/ruCLEVR)** — русская версия набора для визуального рассуждения из MERA-evaluation.  
2. **[ruVQA](https://huggingface.co/datasets/MERA-evaluation/ruVQA)** — русскоязычный датасет для задачи визуальных вопросов и ответов (Visual Question Answering) из MERA-evaluation.  
3. **[MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru)** — русская версия мультимодального бенчмарка MMBench от deepvk.  
4. **[MWS-Vision-Bench](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench)** — датасет для понимания технических диаграмм от MTSAIR.  
5. **[Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)** — набор данных для понимания документов от HuggingFaceM4 (используются только документы с одним изображением). 

Объединение и подготовка данных для обучения Qwen2.5 выполняется в ноутбуке
1.0-data-consolidation.ipynb
Поддерживается схема HuggingFace Datasets для совместимости
Итоговый объединённый датасет опубликован на Kaggle https://www.kaggle.com/datasets/ivkemaev/docvqa-ru-eng-v1

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

--------


