# features.py
from textblob import TextBlob
import re

# Load dynamic vocabularies from external lexicon files
nrc_positive_words = set()
nrc_negative_words = set()
sentiment_w = 0.7
lexicon_w = 0.3

# Load NRC Emotion Lexicon (positive and negative only)
try:
    with open("data/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r") as f:
        for line in f:
            word, emotion, assoc = line.strip().split("\t")
            if assoc == "1":
                if emotion == "positive":
                    nrc_positive_words.add(word)
                elif emotion == "negative":
                    nrc_negative_words.add(word)
except FileNotFoundError:
    print("NRC lexicon not found. Using fallback lists.")
    nrc_positive_words = {"happy", "joyful", "grateful", "excited", "hopeful"}
    nrc_negative_words = {"sad", "angry", "frustrated", "anxious", "miserable"}

def extract_features(message: str) -> dict:
    sentiment = TextBlob(message).sentiment.polarity
    tokens = re.findall(r"\b\w+\b", message.lower())
    contains_positive = any(word in nrc_positive_words for word in tokens)
    contains_negative = any(word in nrc_negative_words for word in tokens)
    lexicon_sentiment = (
        1 if contains_positive and not contains_negative
        else -1 if contains_negative and not contains_positive
        else 0
    )
    # Combine polarity and lexicon sentiment
    combined_sentiment = sentiment_w * sentiment + lexicon_w * lexicon_sentiment
    return {
        "sentiment": sentiment,
        "contains_positive": contains_positive,
        "contains_negative": contains_negative,
        "lexicon_sentiment": lexicon_sentiment,
        "combined_sentiment": combined_sentiment
    }
