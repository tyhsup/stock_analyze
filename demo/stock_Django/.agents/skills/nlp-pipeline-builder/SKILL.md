---
name: nlp-pipeline-builder
description: >
  Build and orchestrate NLP/ML pipelines for text classification, named entity recognition (NER),
  sentiment analysis, text generation, and embeddings using Hugging Face transformers and spaCy.
  Use when: designing NLP pipelines, selecting transformer models, preprocessing text data,
  building text classifiers, extracting named entities, analyzing sentiment, generating text,
  creating embeddings, or when user mentions NLP, natural language processing, text analysis,
  transformers, BERT, GPT, tokenization, or ML pipelines.
---

# NLP Pipeline Builder

Expert guidance for building production-ready Natural Language Processing pipelines.

## Core Competencies

### 1. Text Preprocessing
- Tokenization strategies (word-level, subword BPE, SentencePiece)
- Text normalization (lowercasing, stemming, lemmatization)
- Stop word removal and custom filtering
- Unicode handling and encoding normalization
- Language detection and multi-language support

### 2. Text Classification
- Binary and multi-class classification
- Multi-label classification
- Zero-shot classification with transformer models
- Fine-tuning pre-trained models (BERT, RoBERTa, XLNet)
- Feature extraction and TF-IDF baselines

### 3. Named Entity Recognition (NER)
- Pre-trained NER models (spaCy, Hugging Face)
- Custom entity types and training data annotation
- Token classification with transformers
- Entity linking and disambiguation
- Nested and overlapping entity handling

### 4. Sentiment Analysis
- Document-level and aspect-based sentiment
- Fine-grained sentiment (5-class) vs binary
- Domain-specific sentiment models
- Multilingual sentiment analysis
- Handling sarcasm, negation, and context

### 5. Text Generation
- Causal language models (GPT-2, GPT-Neo)
- Seq2seq models (T5, BART) for summarization
- Controlled text generation with prompts
- Temperature, top-k, top-p sampling strategies
- Beam search vs nucleus sampling trade-offs

### 6. Embeddings & Semantic Search
- Sentence embeddings (Sentence-BERT, all-MiniLM)
- Word embeddings (Word2Vec, GloVe, FastText)
- Contextual embeddings from transformers
- Vector similarity search (cosine, dot product)
- Dimensionality reduction (PCA, UMAP, t-SNE)

## Pipeline Architecture Patterns

### Data Flow
```
Raw Text → Preprocessing → Feature Extraction → Model Inference → Post-processing → Output
```

### Best Practices
1. **Modular design**: Each pipeline stage should be independently testable
2. **Batch processing**: Use `DataLoader` with appropriate batch sizes for GPU efficiency
3. **Caching**: Cache tokenized inputs and model outputs for repeated inference
4. **Error handling**: Graceful fallbacks for edge cases (empty text, unknown languages)
5. **Monitoring**: Log inference latency, throughput, and model confidence scores

## Technology Stack

### Primary Libraries
- `transformers` (Hugging Face) — Pre-trained models and tokenizers
- `datasets` (Hugging Face) — Dataset loading and preprocessing
- `spacy` — Industrial-strength NLP
- `scikit-learn` — ML utilities and baselines
- `torch` / `tensorflow` — Deep learning backends

### Supporting Libraries
- `tokenizers` — Fast tokenization
- `sentencepiece` — Subword tokenization
- `nltk` — Classic NLP utilities
- `pandas` — Data manipulation
- `numpy` — Numerical operations

## Model Selection Guide

| Task | Recommended Models | Notes |
|------|-------------------|-------|
| Text Classification | `bert-base-uncased`, `roberta-base` | Fine-tune on domain data |
| NER | `bert-base-NER`, `spaCy en_core_web_trf` | Consider domain-specific models |
| Sentiment | `nlptown/bert-base-multilingual-uncased-sentiment` | Supports 5-star rating |
| Summarization | `facebook/bart-large-cnn`, `t5-base` | Adjust max_length for output |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Best speed/quality trade-off |
| Zero-Shot | `facebook/bart-large-mnli` | No training data needed |

## Code Patterns

### Pipeline Initialization
```python
from transformers import pipeline

# Quick start with Hugging Face pipelines
classifier = pipeline("text-classification", model="bert-base-uncased")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
sentiment = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
```

### Custom Training Loop
```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_classes
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Embedding Generation
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["This is a sentence.", "This is another sentence."]
embeddings = model.encode(sentences)

# Cosine similarity
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
```

## Performance Optimization

- **Mixed precision**: Use `fp16=True` in TrainingArguments for faster training
- **Gradient accumulation**: Simulate larger batches on limited GPU memory
- **Model distillation**: Use DistilBERT/TinyBERT for production inference
- **ONNX Runtime**: Export models to ONNX for optimized CPU/GPU inference
- **Quantization**: INT8 quantization for edge deployment

## Evaluation Metrics

| Task | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| Classification | Accuracy, F1-score | Precision, Recall, AUC-ROC |
| NER | Entity-level F1 | Token-level accuracy |
| Sentiment | Accuracy, Macro-F1 | Confusion matrix |
| Generation | ROUGE, BLEU | Perplexity, BERTScore |
| Embeddings | Cosine similarity | Retrieval MRR, nDCG |
