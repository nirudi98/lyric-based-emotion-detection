import numpy as np
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from scipy.sparse import hstack, csr_matrix

from scripts.preprocessor.text_preprocessing import load_nrc_lexicon, extract_nrc_features
from scripts.constants import MULTICLASS_ML_ENSEMBLE, MULTICLASS_EMOTION_LYRIC_PATH

PLUTCHIK_EMOTIONS = [ "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
THRESHOLDS = {
    "anger": 0.5,
    "anticipation": 0.5,
    "disgust": 0.75,
    "fear": 0.5,
    "joy": 0.5,
    "sadness": 0.5,
    "surprise": 0.75,
    "trust": 0.5
}

def multilabel_vectorization(emotion):
    vec = np.zeros(len(PLUTCHIK_EMOTIONS), dtype=int)
    if pd.isna(emotion):
        return vec
    for e in emotion.split("|"):
        if e in PLUTCHIK_EMOTIONS:
            vec[PLUTCHIK_EMOTIONS.index(e)] = 1
    return vec

# vectorization and NRC normalization
def build_features(x_train_text, x_test_text):
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=60000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2"
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(x_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(x_test_text)

    # NRC features
    nrc = load_nrc_lexicon()
    X_train_nrc = np.array(extract_nrc_features(x_train_text, nrc))
    X_test_nrc = np.array(extract_nrc_features(x_test_text, nrc))
    # Normalize NRC
    scaler = StandardScaler(with_mean=False)
    X_train_nrc = scaler.fit_transform(X_train_nrc)
    X_test_nrc = scaler.transform(X_test_nrc)

    x_train = hstack([X_train_tfidf, csr_matrix(X_train_nrc)])
    x_test = hstack([X_test_tfidf, csr_matrix(X_test_nrc)])

    return x_train, x_test, tfidf_vectorizer, scaler

# stacking model
def build_stacking_model():
    base_estimators = [
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)),
        ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=10000, C=0.7, random_state=123))),
        ("nb", MultinomialNB(alpha=0.3)),
        ("rf", RandomForestClassifier(n_estimators=50, n_jobs=-1)),
        ("dt", DecisionTreeClassifier(max_depth=20)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]
    meta_learner = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        stack_method="predict_proba",
        cv=5,
        n_jobs=1
    )

# updating the stacking classifier to support multi-label
def build_multilabel_stack():
    return OneVsRestClassifier(build_stacking_model(), n_jobs=1)

def evaluate_multilabel(Y_true, Y_prob, threshold=0.5):
    print("\n--- Evaluation Results ----")
    Y_pred = (Y_prob >= threshold).astype(int)

    print("\nExact-match accuracy:", np.mean(np.all(Y_true == Y_pred, axis=1)))
    print("Micro F1:", f1_score(Y_true, Y_pred, average="micro"))
    print("Macro F1:", f1_score(Y_true, Y_pred, average="macro"))

    print("\nClassification Report:")
    print(classification_report(Y_true, Y_pred, target_names=PLUTCHIK_EMOTIONS, zero_division=0))

    print("\nConfusion Matrix")
    cms = multilabel_confusion_matrix(Y_true, Y_pred)
    for i, label in enumerate(PLUTCHIK_EMOTIONS):
        tn, fp, fn, tp = cms[i].ravel()
        print(f"\n{label.upper()}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

def save_stacked_ml_model(stack, vectors, extra):
    filename = "ml_ensemble.joblib"
    save_path = os.path.join(MULTICLASS_ML_ENSEMBLE, filename)

    joblib.dump({"model": stack, "tfidf": vectors, "extra_objects": extra}, save_path, compress=3)

    print(f"\n---- ML Ensemble Model Saved Successfully at {save_path} ----")

def populate_missing_emotion_nrc(emotion):
    emotion = emotion.copy()
    emotion_map = { "happy": "joy", "sad": "sadness", "angry": "anger" }
    missing_mask = emotion["emotion_NRC"].isna()
    emotion.loc[missing_mask, "emotion_NRC"] = (emotion.loc[missing_mask, "emotion_4Q"].map(emotion_map))
    return emotion

if __name__ == "__main__":
    print("\n---- Training ML Ensemble ----")

    emo_df = pd.read_csv(MULTICLASS_EMOTION_LYRIC_PATH)
    # check for null values compare with 4Q emotions - if can be directly mapped, map
    # emo_df = populate_missing_emotion_nrc(emo_df)
    emo_df = emo_df.dropna(subset=["lyric", "emotion_NRC"])
    X_text = emo_df["lyric"].astype(str)
    Y = np.vstack(emo_df["emotion_NRC"].apply(multilabel_vectorization).values)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, Y,
        test_size=0.2,
        random_state=100
    )
    print("\n---- Building ML Ensemble ----")
   # feature extraction
    X_train, X_test, tfidf, nrc_scaler = build_features(X_train_text, X_test_text)
    model = build_multilabel_stack()
    model.fit(X_train, y_train)
    print("\n---- Saving ML Ensemble ----")
    save_stacked_ml_model(stack=model, vectors=tfidf,
                          extra={ "nrc_scaler": nrc_scaler, "labels": PLUTCHIK_EMOTIONS })
    print("\n---- Predicting Probabilities ML Ensemble ----")
    Y_prob = model.predict_proba(X_test)
    y_pred = np.zeros_like(Y_prob, dtype=int)

    for i, label in enumerate(PLUTCHIK_EMOTIONS):
        y_pred[:, i] = (Y_prob[:, i] >= THRESHOLDS[label]).astype(int)

    evaluate_multilabel(y_test, y_pred)
