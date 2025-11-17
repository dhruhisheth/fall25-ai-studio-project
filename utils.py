"""
Utility functions for the Streamlit dashboard
"""
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
import fsspec
from itertools import islice
from collections import Counter
import re

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Configuration - matching notebook exactly
REPO = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORIES = ["Software", "Video_Games", "All_Beauty"]
ALL_CATEGORIES = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
              "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
              "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific",
              "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
              "Software", "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games",
              "Unknown"]
N_PER_CAT = 1000  # Reduced for faster loading in dashboard
N_META = 60_000


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from a string
    """
    if not isinstance(text, str):
        return ""
    return text.translate(str.maketrans("", "", string.punctuation))


def preprocess_text(text: str) -> str:
    """
    Preprocess text: lowercase, remove punctuation, normalize whitespace
    Matches the exact pipeline from the notebook (Cell 21)
    """
    if not isinstance(text, str):
        return ""
    
    # Exact pipeline from notebook: .str.lower().apply(remove_punctuation).str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.lower()
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text


def create_clean_review(text: str) -> str:
    """
    Creates clean_review exactly as in notebook (Cell 21)
    """
    if not isinstance(text, str):
        return ""
    return preprocess_text(text)


def create_clean_title(title: str) -> str:
    """
    Creates clean_title exactly as in notebook (Cell 21)
    """
    if not isinstance(title, str):
        return ""
    return preprocess_text(title)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using WordNetLemmatizer
    Exact implementation from notebook (Cell 23)
    """
    if not isinstance(text, str):
        return ""
    
    try:
        tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmas)
    except Exception as e:
        return text  # Return original text if lemmatization fails


def tokenize_review(text: str) -> list:
    """
    Tokenize review text using word_tokenize
    Matches notebook Cell 33
    """
    if not isinstance(text, str):
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return []


def create_sentiment_label(rating: float) -> str:
    """
    Create sentiment label from rating
    Exact implementation from notebook (Cell 26)
    - 4-5 stars: positive
    - 3 stars: neutral
    - 1-2 stars: negative
    """
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'


def analyze_sentiment_simple(text: str) -> tuple[str, str]:
    """
    Simple rule-based sentiment analysis
    Returns (sentiment, confidence)
    """
    if not text:
        return 'neutral', 'low'
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = [
        'excellent', 'great', 'good', 'amazing', 'wonderful', 'love', 'perfect',
        'fantastic', 'awesome', 'outstanding', 'brilliant', 'superb', 'best',
        'satisfied', 'happy', 'pleased', 'recommend', 'highly', 'quality'
    ]
    
    # Negative keywords
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed',
        'poor', 'waste', 'broken', 'defective', 'faulty', 'useless',
        'hate', 'regret', 'avoid', 'problem', 'issue', 'complaint'
    ]
    
    # Count occurrences
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        return 'positive', 'medium'
    elif negative_count > positive_count:
        return 'negative', 'medium'
    else:
        return 'neutral', 'low'


def stream_jsonl(url: str, limit: int | None = None):
    """
    Stream a JSONL file line-by-line from Hugging Face
    """
    try:
        with fsspec.open(url, "rt") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                obj = json.loads(line)
                
                if "price" in obj and obj["price"] is not None:
                    obj["price"] = str(obj["price"])
                
                yield obj
    except Exception as e:
        print(f"Error streaming JSONL: {e}")
        return


def load_category_into_review(category: str, n_reviews: int):
    """
    Load one category's reviews as DataFrame
    Exact implementation from notebook (Cell 14)
    """
    try:
        reviews_url = f"hf://datasets/{REPO}/raw/review_categories/{category}.jsonl"
        
        data = (
            {k: row.get(k) for k in ["rating", "title", "text"]}
            for row in islice(stream_jsonl(reviews_url), n_reviews)
        )
        
        reviews_df = pd.DataFrame(data).assign(category=category)
        return reviews_df
    except Exception as e:
        print(f"Error loading category {category}: {e}")
        return pd.DataFrame()


def load_sample_data(n_reviews_per_cat: int = 100):
    """
    Load sample review data for the dashboard
    Matches notebook approach (Cell 15) but with smaller sample size
    """
    try:
        all_reviews = []
        
        # Use CATEGORIES for faster loading (can switch to ALL_CATEGORIES for full dataset)
        for cat in CATEGORIES:
            try:
                r_df = load_category_into_review(cat, n_reviews=n_reviews_per_cat)
                if not r_df.empty:
                    all_reviews.append(r_df)
            except Exception as e:
                print(f"Could not load category {cat}: {e}")
                continue
        
        if all_reviews:
            reviews_df = pd.concat(all_reviews, ignore_index=True)
            
            # Apply the exact preprocessing pipeline from notebook
            # Cell 21: clean_review and clean_title
            reviews_df['clean_review'] = reviews_df['text'].apply(create_clean_review)
            reviews_df['clean_title'] = reviews_df['title'].apply(create_clean_title)
            
            # Cell 24: lemmatized_review and lemmatized_title
            reviews_df['lemmatized_review'] = reviews_df['clean_review'].apply(lemmatize_text)
            reviews_df['lemmatized_title'] = reviews_df['clean_title'].apply(lemmatize_text)
            
            # Cell 27: sentiment_labels
            reviews_df['sentiment_labels'] = reviews_df['rating'].apply(create_sentiment_label)
            
            # Cell 33: tokenized_review
            reviews_df['tokenized_review'] = reviews_df['clean_review'].apply(tokenize_review)
            
            # Ensure required columns exist
            required_cols = ['rating', 'title', 'text', 'category']
            for col in required_cols:
                if col not in reviews_df.columns:
                    reviews_df[col] = None
            
            # Remove rows with missing essential data
            reviews_df = reviews_df.dropna(subset=['text', 'rating'])
            
            return reviews_df
        else:
            # Return mock data if loading fails
            return create_mock_data()
    
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return create_mock_data()


def create_mock_data():
    """
    Create mock review data for demonstration when real data cannot be loaded
    """
    mock_reviews = [
        {
            'rating': 5.0,
            'title': 'Excellent product!',
            'text': 'This product exceeded my expectations. The quality is outstanding and I would highly recommend it to anyone.',
            'category': 'Software',
            'asin': 'MOCK001'
        },
        {
            'rating': 4.0,
            'title': 'Great value for money',
            'text': 'Good product overall. Works as expected and the price is reasonable. Some minor improvements could be made.',
            'category': 'Software',
            'asin': 'MOCK002'
        },
        {
            'rating': 3.0,
            'title': 'Average product',
            'text': 'The product is okay. Nothing special, but it does the job. Could be better.',
            'category': 'Video_Games',
            'asin': 'MOCK003'
        },
        {
            'rating': 2.0,
            'title': 'Disappointed',
            'text': 'Not what I expected. The product has several issues and the quality is poor. Would not recommend.',
            'category': 'Video_Games',
            'asin': 'MOCK004'
        },
        {
            'rating': 1.0,
            'title': 'Terrible experience',
            'text': 'This is the worst product I have ever purchased. It broke immediately and customer service was unhelpful.',
            'category': 'All_Beauty',
            'asin': 'MOCK005'
        },
        {
            'rating': 5.0,
            'title': 'Amazing quality!',
            'text': 'I love this product! It works perfectly and the design is beautiful. Worth every penny.',
            'category': 'All_Beauty',
            'asin': 'MOCK006'
        },
        {
            'rating': 4.0,
            'title': 'Very satisfied',
            'text': 'Great product with good features. Minor issues but overall very happy with the purchase.',
            'category': 'Software',
            'asin': 'MOCK007'
        },
        {
            'rating': 2.0,
            'title': 'Not worth it',
            'text': 'Poor quality and does not work as advertised. Regret buying this product.',
            'category': 'Video_Games',
            'asin': 'MOCK008'
        }
    ]
    
    return pd.DataFrame(mock_reviews)

