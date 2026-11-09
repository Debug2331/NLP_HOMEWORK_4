"""
Part II, Q1: NLP preprocessing pipeline.
Steps: tokenize -> remove stopwords -> stem.
Requires: pip install nltk   (the NLTK data is downloaded automatically below)
"""
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download the NLTK resources the first time the script is run
for resource in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(resource, quiet=True)


def preprocess(sentence):
    """Tokenize, remove stopwords/punctuation, and stem a sentence.
    Returns (tokens, tokens_without_stopwords, stemmed_tokens)."""
    # 1. Tokenize: split the sentence into words and punctuation marks
    tokens = word_tokenize(sentence)

    # 2. Remove English stopwords (compare in lowercase) and punctuation
    stop_words = set(stopwords.words("english"))
    filtered = [t for t in tokens
                if t.lower() not in stop_words and t not in string.punctuation]

    # 3. Stem each remaining word with the Porter stemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in filtered]
    return tokens, filtered, stemmed


if __name__ == "__main__":
    text = "NLP techniques are used in virtual assistants like Alexa and Siri."
    tokens, filtered, stemmed = preprocess(text)
    print("Original Tokens:         ", tokens)
    print("Tokens Without Stopwords:", filtered)
    print("Stemmed Words:           ", stemmed)
