# text preprocessing is included in this script
# normalization | noise removal | tokenization | stopwords removal | lemmatization | fragment text | word embedding

import re
import contractions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from collections import defaultdict

# initialization
wordnet_lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

NOT_USED_STOP_WORDS = {'more', 'aren', "mightn't", 'doesn', 'isn', "didn't", 'wouldn', "won't", 'ain', 'couldn',
                       "shouldn't", "weren't", 'didn', "hadn't", 'needn', 'shouldn', 'mustn', "mustn't", "wasn't",
                       "couldn't", 'wasn', "hasn't", 'very', 'most', 'hadn', "wouldn't", "don't", "aren't", 'hasn',
                       "needn't", "haven't", 'nor', 'no', 'won', 'not', 'never', 'haven', "isn't", 'don', "doesn't"}

ADDITIONAL_STOP_WORDS = {"'s", "'re", "'m", "'ve", "'d", "'ll"}

STOP_WORDS = STOP_WORDS - NOT_USED_STOP_WORDS | ADDITIONAL_STOP_WORDS

BRACKETS_PATTERN = r'\[.*?\]'

_WORDS_IN_FRAGMENT = 60

def normalize_text(text, expand_contractions=True):
    text = text.lower()
    if expand_contractions:
        text = contractions.fix(text)
    return text

def remove_noise(text, remove_brackets=True, remove_punctuation=True):
    if remove_brackets:
        text = re.sub(BRACKETS_PATTERN, '', text)
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    # removing numbers, newlines and spaces
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def tokenize_text(text):
    return word_tokenize(text)

def remove_stop_words(tokens):
    return [w for w in tokens if w not in STOP_WORDS]

def lemmatize_text(tokens):
    return [wordnet_lemmatizer.lemmatize(w) for w in tokens]

def preprocess(
        text,
        expand_contractions=True,
        remove_brackets=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize_words=True
):
    text = normalize_text(text, expand_contractions=expand_contractions)
    text = remove_noise(text, remove_brackets, remove_punctuation)
    tokens = tokenize_text(text)

    if remove_stopwords:
        tokens = remove_stop_words(tokens)

    if lemmatize_words:
        tokens = lemmatize_text(tokens)

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# def remove_stop_words(text):
#     text_without_stop_words = ' '.join([word for word in word_tokenize(text) if word not in STOP_WORDS])
#     text_without_stop_words = re.sub(r'\s+\'\s+', ' ', text_without_stop_words)
#     return text_without_stop_words

# fragmentation is ignored for now

# feature extraction

########################################################################################################################
from collections import Counter
from nltk.tokenize import word_tokenize
from scripts.constants import PREPROCESSED_EMOTION_LYRIC_PATH
from sklearn.preprocessing import LabelEncoder
import os

from scripts.constants import LEXICON_PATH

PLUTCHIK_ORDER = ["anger", "fear", "anticipation", "trust","surprise", "sadness", "joy", "disgust"]
SENTIMENT_ORDER = ["positive", "negative"]

emotion_labels = {"anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"}
sentiment_labels = {"positive", "negative"}

def load_nrc_lexicon():
    file_path = os.path.join(LEXICON_PATH, f"NRC-Emotion-Lexicon-Wordlevel.txt")
    nrc = pd.read_csv(file_path, sep="\t", names=["word", "emotion", "association"])
    print(nrc.head())
    nrc = nrc[(nrc["association"] == 1) & (nrc["emotion"].isin(emotion_labels))]
    print("Loading NRC lexicon...")

    nrc = nrc[["word", "emotion"]]
    nrc["word"] = nrc["word"].str.lower()

    return nrc

def build_nrc_lookup(nrc_df):
    emotion_lookup = defaultdict(set)

    for _, row in nrc_df.iterrows():
        emotion_lookup[row["word"]].add(row["emotion"])
    return emotion_lookup

# function to convert lyrics into NRC emotion vectors
def extract_nrc_features(text_series, nrc_lexicon):
    # emotion lookup
    emotion_word_map = {
        emotion: set(
            nrc_lexicon[
                nrc_lexicon["emotion"] == emotion
            ]["word"]
        )
        for emotion in PLUTCHIK_ORDER
    }

    features = []

    for text in text_series:
        tokens = word_tokenize(text.lower())
        emotion_counts = np.zeros(len(PLUTCHIK_ORDER))

        for i, emotion in enumerate(PLUTCHIK_ORDER):
            emotion_counts[i] = sum(
                token in emotion_word_map[emotion]
                for token in tokens
            )

        # Normalize (important!)
        total = emotion_counts.sum()
        if total > 0:
            emotion_counts = emotion_counts / total

        features.append(emotion_counts)

    return np.array(features)

def build_tfidf_features():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=5,
        max_df=0.85,
        sublinear_tf=True,
    )

def build_feature_matrix(df, vectorizer, emotion_lookup):
    # Text features
    X_text = vectorizer.fit_transform(df["lyric"])
    # NRC features
    X_nrc = np.vstack(df["lyric"].apply(lambda x: extract_nrc_features(x, emotion_lookup)))
    # Combine
    X = hstack([X_text, X_nrc])
    return X

def encode_labels(df):
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["emotion_NRC"])
    return y, encoder

# train and test split
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
