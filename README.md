# 📦 AI Studio Project — Amazon Review Sentiment Analysis (Fall 2025)

This repository contains my end-to-end **Amazon Review Sentiment Analysis** pipeline developed for the **Fall 2025 AI Studio Program**. The project includes full NLP preprocessing, heuristic rules, traditional machine learning models, visualizations, and an interactive **Streamlit dashboard**.

## 🚀 Project Overview
This project classifies Amazon product reviews into **Positive**, **Neutral**, or **Negative** sentiments. It includes:
- Complete preprocessing pipeline (cleaning → lemmatization → tokenization → stopword removal)
- Heuristic scoring (negation, emphasis, sentiment lexicons)
- Machine learning models for classification
- Streamlit dashboard fully connected to backend functions
- Data exploration and visual analytics (word clouds, histograms, sentiment charts)

Future updates will add **BERT**, **RoBERTa**, and **DistilBERT** for transformer-based performance improvements.

## 🎛️ Features

### 🔍 Sentiment Analysis
- Single or batch review analysis  
- Shows each preprocessing step  
- Predicts sentiment with a confidence score  
- Lexicon + heuristic + ML hybrid pipeline

### 📊 Data Exploration
- Load Amazon datasets from HuggingFace  
- Filter by ratings, categories, sentiment  
- Visualizations:  
  - Word clouds  
  - Sentiment distribution  
  - Review-length histograms  
  - Rating vs sentiment breakdown  

### 🧠 Machine Learning Models
- Naive Bayes (Gaussian, Bernoulli, Multinomial)  
- Logistic Regression  
- Support Vector Machine  
- Random Forest  
- **Coming soon:** BERT, DistilBERT, RoBERTa, ABSA

### 🖥️ Streamlit Dashboard
- Amazon-themed UI styling  
- Fully connected to the preprocessing pipeline  
- Real-time prediction + visualization  
- Clean navigation with multiple pages (Home, Analysis, Exploration, About)

## 🛠️ Installation
Clone the repo and install dependencies:

