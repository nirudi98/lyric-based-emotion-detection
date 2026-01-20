import re, os, joblib
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

from scripts.constants import LYRICS_FETCHED, CLEANED_LYRICS_FETCHED, LEXICON_PATH

ERROR_PATTERN = re.compile(r"^Lyrics fetch failed for (.*?) - (.*?):")
PLUTCHIK_EMOTIONS = [ "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
LABEL_THRESHOLDS = {"anger": 0.5, "anticipation": 0.5, "disgust": 0.75,  "fear": 0.5, "joy": 0.5, "sadness": 0.5, "surprise": 0.75, "trust": 0.5}

def clean_lyrics_fetch_log(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            # separate the songs of which lyrics did not fetch properly
            line = line.strip()
            match = ERROR_PATTERN.match(line)
            if match:
                song_title = match.group(1).strip()
                artist = match.group(2).strip()
                outfile.write(f"{song_title} | {artist}\n")

# ML ensemble - evaluate individual models to choose the base learners for ML+DL ensemble
def choose_best_ml_models(models, X, Y, save_path):
    print("\n---- Evaluating Individual ML Base Models ----\n ----")
    kf = KFold(n_splits=5, shuffle=True, random_state=150)
    scores = {}

    for name, base_model in models.items():
        print(f"\n Base Model: {name} ---> ")
        Y_oof = np.zeros_like(Y, dtype=float)

        for tr, va in kf.split(X):
            clf = OneVsRestClassifier(base_model, n_jobs=1)
            clf.fit(X[tr], Y[tr])
            Y_oof[va] = clf.predict_proba(X[va])

        Y_pred = np.zeros_like(Y_oof, dtype=int)
        for i, label in enumerate(PLUTCHIK_EMOTIONS):
            Y_pred[:, i] = (Y_oof[:, i] >= LABEL_THRESHOLDS[label]).astype(int)

        micro = f1_score(Y, Y_pred, average="micro")
        macro = f1_score(Y, Y_pred, average="macro")

        scores[name] = {"micro_f1": micro, "macro_f1": macro}

        print(f"Micro F1: {micro:.4f}")
        print(f"Macro F1: {macro:.4f}")

        # Freeze model (fit on full data)
        final_clf = OneVsRestClassifier(base_model, n_jobs=1)
        final_clf.fit(X, Y)

        model_path = os.path.join(save_path, f"{name}_ml_model.joblib")
        joblib.dump(final_clf, model_path, compress=3)

        print(f"Saved frozen model → {model_path}")

    return scores

def load_lexicon_nrc():
    file_path = os.path.join(LEXICON_PATH, "NRC-Emotion-Lexicon-Wordlevel.txt")
    nrc = pd.read_csv(file_path, sep="\t", names=["word", "emotion", "association"])

    nrc = nrc[ (nrc["association"] == 1) & (nrc["emotion"].isin(PLUTCHIK_EMOTIONS))]
    nrc["word"] = nrc["word"].str.lower()
    return nrc


def build_nrc_emotions_map(nrc_df):
    emotion_map = defaultdict(set)
    for _, row in nrc_df.iterrows():
        emotion_map[row["emotion"]].add(row["word"])
    return emotion_map


def extract_nrc_features_process(text_series, emotion_word_map):
    features = []

    for text in text_series:
        tokens = word_tokenize(text.lower())
        emotion_counts = np.zeros(len(PLUTCHIK_EMOTIONS))

        for i, emotion in enumerate(PLUTCHIK_EMOTIONS):
            emotion_counts[i] = sum(
                token in emotion_word_map[emotion]
                for token in tokens
            )

        total = emotion_counts.sum()
        if total > 0:
            emotion_counts /= total

        features.append(emotion_counts)
    return np.array(features)



if __name__ == "__main__":
    clean_lyrics_fetch_log(LYRICS_FETCHED, CLEANED_LYRICS_FETCHED)
    print("---- Log file cleaned successfully ----")
