import pandas as pd
import os
import re

from scripts.constants import SONG_DATASET_PATH, PREPROCESSED_LYRIC_PATH, MULTICLASS_EMOTION_LYRIC_PATH
from scripts.constants import LEXICON_PATH

from collections import Counter

from scripts.preprocessor.text_preprocessing import preprocess

LYRICS_COLUMN_NAME = "lyrics"
PLUTCHIK_EMOTIONS = { "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust" }
FOURQ_TO_PLUTCHIK = {
    "happy": {"joy", "trust", "anticipation"},
    "sad": {"sadness", "fear"},
    "angry": {"anger", "disgust"},
    "relaxed": {"trust", "joy", "anticipation"}
}

SENTIMENT_LABELS = { "positive", "negative" }

# separate lexicon extension code
def complete_nrc_lexicon_load():
    file_path = os.path.join(LEXICON_PATH, f"NRC-Emotion-Lexicon-Wordlevel.txt")
    nrc = pd.read_csv(file_path, sep="\t", names=["word", "emotion", "association"])

    nrc = nrc[nrc["association"] == 1].drop(columns="association")
    emotion_labels = {"anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"}
    sentiment_labels = {"positive", "negative"}

    nrc["type"] = nrc["emotion"].apply(lambda x: "sentiment" if x in sentiment_labels else "emotion")
    nrc["sentiment"] = nrc["emotion"].apply(lambda x: x if x in sentiment_labels else None)
    nrc["emotion_label"] = nrc["emotion"].apply(lambda x: x if x in emotion_labels else None)

    return nrc

# scoring for the above lexicon way
def complete_score_lyrics_with_nrc(text, nrc):
    words = re.findall(r"[\w']+", text.lower())
    matched = nrc[nrc["word"].isin(words)]

    emotion_counts = Counter(matched["emotion_label"].dropna())
    sentiment_counts = Counter(matched["sentiment"].dropna())

    total_emotions = sum(emotion_counts.values())
    total_sentiments = sum(sentiment_counts.values())

    emotion_scores = (
        {e: round(c / total_emotions, 4) for e, c in emotion_counts.items()}
        if total_emotions > 0 else {}
    )

    sentiment_scores = (
        {s: round(c / total_sentiments, 4) for s, c in sentiment_counts.items()}
        if total_sentiments > 0 else {}
    )

    return emotion_scores, sentiment_scores

def top_n_emotion_extraction(emotion_scores, emotion_4q):
    # top emotions above the min threshold defined are returned
    # Case 1: NRC emotions exist → select top-2 by score
    if isinstance(emotion_scores, dict) and len(emotion_scores) > 0:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_emotions) >= 2:
            return sorted_emotions[0][0], sorted_emotions[1][0]
        else:
            return sorted_emotions[0][0], sorted_emotions[0][0]

    fallback = FOURQ_TO_PLUTCHIK.get(emotion_4q)
    if fallback:
        fallback_list = list(fallback)  # <-- FIX
        if len(fallback_list) >= 2:
            return fallback_list[0], fallback_list[1]
        else:
            return fallback_list[0], fallback_list[0]
    # Case 3: fully unresolved (e.g. relaxed)
    return "unknown", "unknown"

def dominant_sentiment(emotion_scores):
    if not emotion_scores:
        return pd.Series([None, 0.0])

    label, score = max(emotion_scores.items(), key=lambda x: x[1])
    return pd.Series([label, score])

def combine_multi_emotions(row):
    e1 = row["emotion_1"]
    e2 = row["emotion_2"]

    if pd.isna(e1) and pd.isna(e2):
        return None

    if pd.isna(e2) or e1 == e2:
        return e1

    return f"{e1}|{e2}"

if __name__ == '__main__':
    print(f"\n---- Extend 4Q Emotion to 8Q Multi-Class Emotion Lexicon ----")
    # when mapping 8Q emotions to a single dominant label this can affect the dataset
    # NRC only gives distributions and not hard labels

    # analyzing the existing filtered dataset
    print(f"\n---- Analyze Filtered Dataset ----")
    song_df = pd.read_csv(SONG_DATASET_PATH)
    song_df.info()

    song_df = song_df.drop(columns=["dataset"])
    # print("Columns in filtered song dataset : ", song_df.columns)
    # print("Shape of filtered song dataset : ", song_df.shape)

    # print("Stats of filtered song dataset", song_df.describe())

    # filtering out the most important columns
    if LYRICS_COLUMN_NAME not in song_df.columns:
        raise ValueError("LYRICS_COLUMN_NAME not found in dataset.")

    lyrics = song_df[LYRICS_COLUMN_NAME]
    song_df = (song_df.dropna(subset=["lyrics"]).reset_index(drop=True))
    # print("Shape of filtered song dataset after dropping null lyrics : ", song_df.shape)
    # preprocess lyrics
    print(f"\n---- Text Preprocessing applied to Lyrics ----")
    song_df["lyrics_clean"] = lyrics.apply(preprocess)

    df_preprocessed = song_df[["lyrics_clean", "emotion_4Q", "emotion_2Q", "genre"]]
    df_preprocessed.to_csv(PREPROCESSED_LYRIC_PATH, index=False)

    # extending NRC Lexicon to support 8 emotions
    print(f"\n---- Extending Emotion Lexicon Supporting 2 Emotions ----")
    df_lyric_preprocessed = pd.read_csv(PREPROCESSED_LYRIC_PATH)
    # check if there are any null rows
    null_rows = df_lyric_preprocessed[df_lyric_preprocessed['lyrics_clean'].isna()]
    print("Null lyrics rows available : ", len(null_rows))
    # remove the null row
    df_lyric_preprocessed = df_lyric_preprocessed.dropna(subset=['lyrics_clean']).reset_index(drop=True)
    print("\nsong list available before extending the emotion: ", len(df_lyric_preprocessed))

    # emotion words against 8 lexicons
    nrc_lexicon = complete_nrc_lexicon_load()
    results = df_lyric_preprocessed["lyrics_clean"].apply(lambda x: complete_score_lyrics_with_nrc(x, nrc_lexicon))
    df_lyric_preprocessed["nrc_emotions"] = results.apply(lambda x: x[0])
    df_lyric_preprocessed["nrc_sentiments"] = results.apply(lambda x: x[1])

    df_lyric_preprocessed[["emotion_1", "emotion_2"]] = (
        df_lyric_preprocessed.apply(
            lambda row: pd.Series(
                top_n_emotion_extraction(row["nrc_emotions"], row["emotion_4Q"])
            ),
            axis=1
        )
    )

    df_lyric_preprocessed[["sentiment_label", "sentiment_score"]] = (df_lyric_preprocessed["nrc_sentiments"].apply(
        dominant_sentiment
    ))
    df_lyric_preprocessed["emotion_NRC"] = df_lyric_preprocessed.apply(combine_multi_emotions, axis=1)
    print("\nsong list available after extending the emotion: ", len(df_lyric_preprocessed))

    updated_nrc_df = df_lyric_preprocessed[[
        "lyrics_clean",
        "genre",
        "emotion_4Q",
        "sentiment_label",
        "sentiment_score",
        "emotion_NRC",
    ]].rename(columns={
        "lyrics_clean": "lyric",
        "genre": "genre",
        "sentiment_label": "sentiment",
    })

    updated_nrc_df.to_csv(MULTICLASS_EMOTION_LYRIC_PATH, index=False)
    print(updated_nrc_df.shape)