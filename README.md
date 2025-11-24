# 📦 AI Studio Project — Amazon Review Sentiment Analysis (Fall 2025)

This repository contains my end‑to‑end **Amazon Review Sentiment Analysis System**, developed for the **Fall 2025 AI Studio Program**. It includes a complete NLP preprocessing pipeline, heuristic rules, multiple transformer-based sentiment models (BERT, DistilBERT, RoBERTa), ABSA (Aspect-Based Sentiment Analysis), evaluation workflows, and a fully interactive Streamlit dashboard.

---

# 🎯 Project Overview

The goal of this project is to build a system that can:

* Clean and preprocess Amazon review text
* Classify sentiment using rule‑based and transformer-based models
* Extract aspect-level sentiment
* Visualize insights using an interactive Streamlit dashboard
* Provide reproducible notebooks for model training
* Deploy the final dashboard publicly

---

# 🚀 Objectives & Goals

### ✔ Build complete text‑processing pipeline

### ✔ Train and evaluate multiple ML & transformer models

### ✔ Implement ABSA for fine‑grained insights

### ✔ Visualize data & results interactively

### ✔ Deploy the final dashboard on Render

---

# 🧠 Methodology

## **1. Data Collection & Exploration**

* Used Amazon Product Reviews dataset
* Explored distribution of ratings, categories, and review lengths
* Identified noise patterns (emojis, special characters, URLs)

## **2. Preprocessing Pipeline**

* Lowercasing
* Lemmatization
* Tokenization
* Stopword removal
* Negation handling
* Normalization rules

## **3. Rule-Based Sentiment Model**

* Keyword-based lexicon
* Polarity scoring
* Negation reversal
* Confidence assignment (low/medium/high)

## **4. Transformer-Based Models**

### Pretrained Models

* **BERT-base-cased**
* **DistilBERT emotion model**
* **Twitter RoBERTa sentiment model**

### Fine-Tuning Experiments

* Fine‑tuned BERT on Amazon review sentiment labels
* Achieved improved accuracy over baseline heuristics

### ABSA (Aspect-Based Sentiment Analysis)

* Implemented using pyABSA
* Extracts aspect terms and their polarity (positive/neutral/negative)

---

# 📊 Results & Key Findings

* Transformer models outperform rule‑based approaches by a wide margin
* ABSA provides richer, more actionable insights for product teams
* RoBERTa performs strongly on short reviews
* BERT fine‑tuning provides most stable performance across categories
* Preprocessing significantly increases model accuracy

---

# 📈 Visualizations Included

* Sentiment distribution plots
* Review length histograms
* Word clouds
* ABSA aspect polarity charts
* Batch analysis sentiment pie charts

---

# 🖥 Streamlit Dashboard

The dashboard provides:

* **Single/Batch review sentiment analysis**
* **Model selection:**

  * Rule‑Based
  * BERT (Pretrained/Fine‑tuned)
  * RoBERTa
* **Confidence scores**
* **Processed text previews**
* **Dataset exploration tools**

Live Deployment:
[https://amazon-sentiment-dashboard.onrender.com/](https://amazon-sentiment-dashboard.onrender.com/)

---

# 🛠 Installation Instructions

## **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/AI-Studio-Project.git
cd AI-Studio-Project
```

## **2. Create Virtual Environment (Optional)**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

## **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Dashboard Locally

```bash
streamlit run app.py
```

Your app will open at:
[http://localhost:8501](http://localhost:8501)

---

# 🤖 Model Training & Evaluation

All training notebooks are included in the following branches:

* `Bert-base-cased-model`
* `Distilbert-base-uncased-emotion`
* `twitter-roberta-base-sentiment`
* `absa`

Each notebook includes:

* Dataset loading
* Tokenization
* Model training
* Evaluation & metrics
* Exporting model checkpoints

---

# 🔍 How to Train Your Own Model

1. Open any notebook (e.g., `Cadence_2A_Bert_base_cased_model.ipynb`)
2. Run preprocessing
3. Fine‑tune the transformer model
4. Save checkpoint to: `models/your-model/`
5. In the Streamlit app → Select **BERT (Fine‑tuned)** → Enter your checkpoint path

---

# 🌐 Deployment

This project can be deployed using Render.

## **Render Deployment (Used for this project)**

`render.yaml` is already included.
Render automatically detects:

* Python environment
* Start command
* Build command

To deploy:

1. Push code to GitHub
2. Go to [https://dashboard.render.com](https://dashboard.render.com)
3. New → Blueprint
4. Select repo
5. Deploy

Your deployed app:
[https://amazon-sentiment-dashboard.onrender.com/](https://amazon-sentiment-dashboard.onrender.com/)

---

# 📚 Project Structure

```
AI-Studio-Project/
│
├── app.py                # Streamlit Dashboard
├── utils.py              # Preprocessing & ML utilities
├── requirements.txt      # Dependencies
├── render.yaml           # Deployment config
├── Dockerfile            # Containerization
├── .dockerignore
│
├── notebooks/ (various branches)
│   ├── BERT fine-tuning
│   ├── RoBERTa sentiment
│   └── ABSA models
│
└── README.md             # Project documentation
```

---

# 👩‍💻 Individual Contributions

**Dhruhi Sheth**

* Developed full preprocessing pipeline
* Implemented rule‑based sentiment engine
* Ran transformer fine‑tuning experiments
* Built and styled complete Streamlit dashboard
* Integrated multiple models (BERT, DistilBERT, RoBERTa)
* Added ABSA capability
* Deployed final project on Render
* Wrote full documentation & README

---

# 🔮 Future Enhancements (Optional)

* Multi‑language support
* Add topic modeling
* Add summarization for long reviews
* Add confusion matrices in dashboard
* Implement vector search for similar reviews

---

# 📜 License

This project is licensed under the **MIT License**.
