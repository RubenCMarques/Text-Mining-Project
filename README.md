# Financial Tweet Sentiment Classification

## Introduction
This project applies modern NLP and machine learning techniques to classify financial tweets as *Bearish*, *Bullish*, or *Neutral*. The goal is to build an automated, scalable sentiment analysis pipeline for short, noisy social media texts. The workflow covers data exploration, preprocessing, feature engineering, and benchmarking of both classical and deep learning models.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)

---

## Project Overview
Financial sentiment analysis provides real-time insight into market mood and supports investment decisions. This project develops an end-to-end classification pipeline—from data exploration and cleaning, through feature engineering, to model evaluation. The project emphasizes scalability and reproducibility, leveraging both classical and deep learning models.

**Key goals:**
- Rigorous preprocessing of short, noisy tweet texts
- Experimentation with multiple text representation techniques (BoW, TF-IDF, Word2Vec, Transformers)
- Benchmarking classical ML, neural networks, and fine-tuned transformers
- Evaluation using macro-averaged F1-score (class imbalance aware)

---

## Dataset
- **Source:** Public financial tweets dataset (~12,000 labeled tweets)
- **Classes:** Bearish, Bullish, Neutral
- **Observations:**
    - Typical tweet: ~86 characters, 12 words
    - Heavy noise: stopwords, URLs, RTs, "co", etc.
    - Frequent bigrams reflect both financial context ("hedge funds", "earnings call") and syndicated content
    - Data is imbalanced (more Neutral tweets)

---

## Methodology

**Exploratory Data Analysis (EDA):**
- Analyzed tweet length, token distributions, word clouds, bigrams
- Identified heavy noise and class imbalance

**Preprocessing:**
- Lowercasing, regex-based cleaning (URLs, RT, mentions, hashtags, HTML, digits, punctuation)
- Tokenization, stopword removal, lemmatization
- Filtered short/noisy tokens

**Feature Engineering:**
- **Bag-of-Words (BoW):** Binary presence/absence vectors
- **TF-IDF:** Weighted word frequency with domain focus
- **Word2Vec:** Custom embeddings; averaged for ML, sequences for LSTM
- **DistilBERT:** [CLS] pooled embeddings, token-level sequences for LSTM and finetuning
- **Sentence-BERT:** Dense sentence embeddings fine-tuned for finance

**Modeling:**
- **KNN:** Baseline, grid-searched across all features
- **Logistic Regression:** Strong performance with TF-IDF/BoW
- **MLP:** Nonlinear, performed best on TF-IDF
- **LSTM:** Sequential modeling with Word2Vec, transformer, and SBERT features
- **Transformers (DistilBERT & FinancialBERT):** Fine-tuned on tweets with PyTorch & Hugging Face

**Evaluation:**
- Macro-averaged Precision, Recall, F1-Score (focus on F1 due to class imbalance)
- 80/20 stratified train/val split; held-out test set for inference

---

## Results

| Model                   | Feature        | Val F1-Score |
|-------------------------|---------------|--------------|
| KNN                     | BoW           | 0.47         |
| KNN                     | DistilBERT    | 0.56         |
| Logistic Regression     | TF-IDF        | 0.64         |
| MLP                     | TF-IDF        | 0.69         |
| LSTM                    | Word2Vec      | 0.55         |
| LSTM                    | Transformer   | 0.62         |
| **DistilBERT (Finetuned)** | Transformer | **0.75**     |

- Fine-tuned DistilBERT achieved the highest macro F1 (0.75), outperforming all classical approaches.
- Classical ML (Logistic Regression, MLP) with TF-IDF/BoW provided strong, interpretable baselines (~0.69 F1).
- Word2Vec improved in sequential LSTM setups.
- Sentence-BERT provided robust embeddings but was outperformed by finetuned transformers.

---

## Future Work
- Integrate additional financial news/context
- Data augmentation for minority classes
- Deploy real-time tweet sentiment API
- Build model explainability tools for analysts

---
