# CS5760 Natural Language Processing – Home Assignment 4 (Fall 2026)

**Student name:** Meka Preetam Reddy
**Roll No. (700#):** 700794858
**Course:** CS5760 Natural Language Processing, University of Central Missouri

This repository contains the Part II (Programming) solutions of Home Assignment 4.
The Part I short answers and the written answers for Part II are in the submitted Word document.

## Files

| File | Description |
|------|-------------|
| `q1_nlp_preprocessing.py` | Q1 – NLP preprocessing pipeline: tokenization, stopword removal, Porter stemming (NLTK) |
| `q2_ner_spacy.py` | Q2 – Named Entity Recognition with spaCy (entity text, label, start/end character offsets) |
| `q3_scaled_dot_product_attention.py` | Q3 – Scaled dot-product attention implemented with NumPy |
| `requirements.txt` | Python packages needed |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

(NLTK data such as `punkt` and `stopwords` is downloaded automatically by `q1_nlp_preprocessing.py`.)

## How to run

```bash
python q1_nlp_preprocessing.py
python q2_ner_spacy.py
python q3_scaled_dot_product_attention.py
```

## Q1 – NLP preprocessing

Sentence: `"NLP techniques are used in virtual assistants like Alexa and Siri."`

1. `word_tokenize` splits the sentence into words and punctuation.
2. Stopwords (NLTK English list) and punctuation are removed (comparison is done in lowercase).
3. `PorterStemmer` reduces each remaining word to its stem.

Output:

```
Original Tokens:          ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']
Stemmed Words:            ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

Note that stems such as `techniqu` are not real words; this is normal for a rule-based stemmer.

## Q2 – Named Entity Recognition (spaCy `en_core_web_sm`)

```
Entity                Label     Start   End
LeBron James          PERSON    1       13
NBA                   ORG       21      24
the United States     GPE       35      52
Laker                 GPE       56      61
MVP                   ORG       84      87
2020                  DATE      91      95
```

The input sentence starts with a space, so offsets are counted from the string exactly as given.
The small spaCy model is not perfect: it labels "Laker" as a location (GPE) although it refers to a team,
and it finds "MVP" as an organization rather than the full phrase "NBA Final MVP".

## Q3 – Scaled dot-product attention

`Attention(Q, K, V) = softmax(Q Kᵀ / √d) V` with d = 4 (key dimension).

```
Attention weights (after softmax):
 [[0.7311 0.2689]
 [0.2689 0.7311]]

Final output matrix:
 [[2.0758 3.0758 4.0758 5.0758]
 [3.9242 4.9242 5.9242 6.9242]]
```

Each query is most similar to its own key (score 2/√4 = 1 versus 0), so each token puts about 73% of its
attention on itself and 27% on the other token.
