import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Dict
from nltk.tokenize import word_tokenize
from collections import Counter

from scripts.constants import PREPROCESSED_EMOTION_LYRIC_PATH

def plot_distribution(distribution: Dict[str, float], title: str, xlabel: str, ylabel: str, rotate_x: bool = True):
    labels = list(distribution.keys())
    values = list(distribution.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_x:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def plot_normalized_distribution(df: pd.DataFrame, column: str, title: str):
    normalized_counts = df[column].value_counts(normalize=True).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(normalized_counts.index, normalized_counts.values)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Proportion")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_dominant_label(df: pd.DataFrame, column: str, title: str):
    counts = df[column].value_counts()
    dominant_label = counts.idxmax()

    colors = ["tab:green" if label == dominant_label else "tab:blue"
        for label in counts.index
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values, color=colors)
    plt.title(f"{title} (Dominant: {dominant_label})")
    plt.xlabel(column)
    plt.ylabel("Count")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_class_imbalance(df: pd.DataFrame, column: str):
    counts = df[column].value_counts()
    min_class = counts.min()
    max_class = counts.max()

    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values)
    plt.title(
        f"Class Imbalance in {column}\n"
        f"Min: {min_class} | Max: {max_class}"
    )
    plt.xlabel(column)
    plt.ylabel("Samples")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def visualization(df):
    plot_normalized_distribution(df, column="genre", title="Genre Distribution")
    plot_dominant_label(df, column="sentiment", title="Sentiment Dominant Label")
    plot_dominant_label(df, column="emotion_NRC", title="Emotion Dominant Label")

    plot_class_imbalance(df, column="emotion_NRC")

def generate_wordcloud_from_series(
    text_series: pd.Series,
    title: str,
    max_words: int = 200,
    width: int = 1000,
    height: int = 500
):
    combined_text = " ".join(text_series.dropna().astype(str))

    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        collocations=False
    ).generate(combined_text)

    plt.figure(figsize=(14, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

def extract_emotion_words(df: pd.DataFrame, emotion_column: str, text_column: str, emotions: list, top_k: int = 50):
    emotion_word_map = {}

    for emotion in emotions:
        emotion_texts = df[df[emotion_column] == emotion][text_column]

        tokens = []
        for text in emotion_texts.dropna():
            tokens.extend(word_tokenize(text))

        counter = Counter(tokens)
        emotion_word_map[emotion] = dict(counter.most_common(top_k))

    return emotion_word_map

def plot_emotion_wordcloud(emotion_word_map: dict):
    for emotion, word_freqs in emotion_word_map.items():
        if not word_freqs:
            continue

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud – {emotion.capitalize()}")
        plt.show()

if __name__ == "__main__":
    print(f"\n---- Emotion Lyric Analysis ----")
    emotion_lyric_df = pd.read_csv(PREPROCESSED_EMOTION_LYRIC_PATH)

    print(emotion_lyric_df.columns.tolist())
    print(emotion_lyric_df.columns.tolist())

    # null values
    null_sum = emotion_lyric_df.isnull().sum().to_dict()
    print("Null values per column:", null_sum)

    # duplicate values
    duplicates = int(emotion_lyric_df.duplicated(subset=["lyric"]).sum())
    print(f"\nDuplicate lyrics count: {duplicates}")
    # if there are duplicate lyrics - duplicate removal - but as there is none no need

    # wordcloud analysis
    generate_wordcloud_from_series(emotion_lyric_df["lyric"], "Word Cloud - Entire Dataset")

    # emotion specific words
    PLUTCHIK_EMOTIONS = ["anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"]
    emotion_words = extract_emotion_words(
        df=emotion_lyric_df,
        emotion_column="emotion_NRC",
        text_column="lyric",
        emotions=PLUTCHIK_EMOTIONS,
        top_k=50
    )
    # plot_emotion_wordcloud(emotion_words)

    # value distributions
    genre_distribution = emotion_lyric_df["genre"].value_counts().to_dict()
    sentiment_distribution = emotion_lyric_df["sentiment"].value_counts().to_dict()
    emotion_4Q_distribution = emotion_lyric_df["emotion_4Q"].value_counts().to_dict()
    emotion_NRC_distribution = emotion_lyric_df["emotion_NRC"].value_counts().to_dict()

    genre_distribution_norm = (emotion_lyric_df["genre"].value_counts(normalize=True).round(4).to_dict())
    sentiment_distribution_norm = (emotion_lyric_df["sentiment"].value_counts(normalize=True).round(4).to_dict())
    emotion_4Q_distribution_norm = (emotion_lyric_df["emotion_4Q"].value_counts(normalize=True).round(4).to_dict())
    emotion_NRC_distribution_norm = (emotion_lyric_df["emotion_NRC"].value_counts(normalize=True).round(4).to_dict())

    dominant_genre = emotion_lyric_df["genre"].value_counts().idxmax()
    dominant_sentiment = emotion_lyric_df["sentiment"].value_counts().idxmax()
    dominant_emotion_4Q = emotion_lyric_df["emotion_4Q"].value_counts().idxmax()
    dominant_emotion_NRC = emotion_lyric_df["emotion_NRC"].value_counts().idxmax()

    print(f"\n---- Genre Distribution ----")
    print(genre_distribution)

    print(f"\n---- Sentiment Distribution ----")
    print(sentiment_distribution)

    print(f"\n---- 8-Emotion (NRC / PLUTCHIK) Distribution ----")
    print(emotion_NRC_distribution)

    print(f"\n---- Dominant Labels ----")
    print(f"Genre: {dominant_genre}")
    print(f"Sentiment: {dominant_sentiment}")
    print(f"Emotion (4Q): {dominant_emotion_4Q}")
    print(f"Emotion (NRC): {dominant_emotion_NRC}")

    # visualization(emotion_lyric_df)

    class_counts = emotion_lyric_df["emotion_NRC"].value_counts()
    print(class_counts)

    # when the full dataset is checked look for oversampling and undersampling fixes
