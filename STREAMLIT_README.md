# Streamlit Dashboard Setup Guide

## Overview
This Streamlit dashboard provides a user-friendly interface for analyzing Amazon product reviews and performing sentiment analysis.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data (if not already downloaded):**
   The dashboard will automatically download required NLTK data on first run, but you can also do it manually:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## Running the Dashboard

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard:**
   The dashboard will open in your default web browser at `http://localhost:8501`

## Features

### 🏠 Home
- Overview of the dashboard features
- Quick statistics

### 🔍 Sentiment Analysis
- **Single Review**: Analyze individual reviews with sentiment classification
- **Batch Analysis**: Process multiple reviews at once
- View processed text and sentiment results

### 📈 Data Exploration
- Load sample review data
- Filter by category and rating
- View detailed review information
- Explore dataset structure

### 📊 Statistics
- Comprehensive visualizations
- Sentiment distribution charts
- Rating analysis
- Category-wise statistics
- Text length analysis

### ⚙️ About
- Project information
- Team members
- Data source details

## Data Loading

The dashboard supports two modes:

1. **Sample Data**: Loads a small sample from the Amazon Reviews dataset (default)
2. **Full Dataset**: Loads data from HuggingFace (requires authentication)

For demonstration purposes, the dashboard includes mock data that will be used if the real data cannot be loaded.

## Notes

- The dashboard uses rule-based sentiment analysis for demonstration
- For production use, integrate trained models from the other branches (e.g., `Bert-base-cased-model`, `twitter-roberta-base-sentiment`)
- Full dataset loading requires HuggingFace authentication

## Troubleshooting

- **NLTK download errors**: Ensure you have internet connection for first-time setup
- **Data loading issues**: The dashboard will fall back to mock data if real data cannot be loaded
- **Port already in use**: Change the port with `streamlit run app.py --server.port 8502`

