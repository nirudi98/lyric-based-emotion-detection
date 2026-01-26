import pandas as pd
import os
import re

from scripts.constants import SONG_DATASET_PATH, PREPROCESSED_LYRIC_PATH, PREPROCESSED_EMOTION_LYRIC_PATH
from scripts.constants import PREPROCESSED_DATASET_PATH, LEXICON_PATH

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

    emotion_counts = Counter(matched["emotion"].dropna())
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

def dominant_label_with_score(scores, label_set):
    filtered = {k: v for k, v in scores.items() if k in label_set}

    if not filtered or max(filtered.values()) == 0:
        return None, 0.0

    label = max(filtered, key=filtered.get)
    return label, filtered[label]

if __name__ == '__main__':
    print(f"\n---- Extend 4Q Emotion to 8Q ----")

    # analyzing the existing filtered dataset
    print(f"\n---- Analyze Filtered Dataset ----")
    # song_df = pd.read_csv(SONG_DATASET_PATH)
    # # song_df.info()
    #
    # song_df = song_df.drop(columns=["dataset"])
    # print("Columns in filtered song dataset : ", song_df.columns)
    # print("Shape of filtered song dataset : ", song_df.shape)

    # print("Stats of filtered song dataset", song_df.describe())

    # filtering out the most important columns
    # if LYRICS_COLUMN_NAME not in song_df.columns:
    #     raise ValueError("LYRICS_COLUMN_NAME not found in dataset.")
    #
    # lyrics = song_df[LYRICS_COLUMN_NAME]
    # # preprocess lyrics
    # print(f"\n---- Text Preprocessing applied to Lyrics ----")
    # song_df["lyrics_clean"] = lyrics.apply(preprocess)
    #
    # df_preprocessed = song_df[["lyrics_clean", "emotion_4Q", "emotion_2Q", "genre"]]
    # df_preprocessed.to_csv(PREPROCESSED_LYRIC_PATH, index=False)

    # # extending NRC Lexicon to support 8 emotions
    print(f"\n---- Extending Emotion Lexicon ----")
    df_lyric_preprocessed = pd.read_csv(PREPROCESSED_LYRIC_PATH)
    # check if there are any null rows
    null_rows = df_lyric_preprocessed[df_lyric_preprocessed['lyrics_clean'].isna()]
    print("Null lyrics rows available : ", len(null_rows))
    # remove the null row
    df_lyric_preprocessed = df_lyric_preprocessed.dropna(subset=['lyrics_clean']).reset_index(drop=True)

    # emotion words against 8 lexicons
    nrc_lexicon = complete_nrc_lexicon_load()
    results = df_lyric_preprocessed["lyrics_clean"].apply(lambda x: complete_score_lyrics_with_nrc(x, nrc_lexicon))
    df_lyric_preprocessed["nrc_emotions"] = results.apply(lambda x: x[0])
    df_lyric_preprocessed["nrc_sentiments"] = results.apply(lambda x: x[1])

    df_lyric_preprocessed[["plutchik_emotion", "plutchik_score"]] = (
        df_lyric_preprocessed["nrc_emotions"].apply(lambda x: pd.Series(dominant_label_with_score(x, PLUTCHIK_EMOTIONS)))
    )

    df_lyric_preprocessed[["sentiment_label", "sentiment_score"]] = (
        df_lyric_preprocessed["nrc_emotions"].apply(lambda x: pd.Series(dominant_label_with_score(x, SENTIMENT_LABELS)))
    )

    updated_nrc_df = df_lyric_preprocessed[[
        "lyrics_clean",
        "genre",
        "emotion_4Q",
        "plutchik_emotion",
        "plutchik_score",
        "sentiment_label",
        "sentiment_score"
    ]].rename(columns={
        "lyrics_clean": "lyric",
        "genre": "genre",
        "plutchik_emotion": "emotion_NRC",
        "sentiment_label": "sentiment",
        "plutchik_score": "emotion_score",
    })

    updated_nrc_df.to_csv(PREPROCESSED_EMOTION_LYRIC_PATH, index=False)
    print(updated_nrc_df.shape)



