import os
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix

from scipy.sparse import hstack, csr_matrix

import tensorflow as tf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import LinearSVC
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense,
    Dropout, Conv1D, GlobalMaxPooling1D, GRU
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scripts.constants import MULTICLASS_EMOTION_LYRIC_PATH, MULTICLASS_HYBRID_MODEL, MULTICLASS_HYBRID_ML_MODEL, \
    MULTICLASS_HYBRID_DL_MODEL
from scripts.utils import load_lexicon_nrc, build_nrc_emotions_map, extract_nrc_features_process

MAX_WORDS = 50000
MAX_LEN = 300
BATCH_SIZE = 64
EPOCHS = 8
N_SPLITS = 5
THRESHOLD = 0.5

PLUTCHIK_EMOTIONS = [ "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust" ]
LABEL_THRESHOLDS = {"anger": 0.5, "anticipation": 0.5, "disgust": 0.75,  "fear": 0.5, "joy": 0.5, "sadness": 0.5, "surprise": 0.75, "trust": 0.5}


def apply_top_k_with_threshold(Y_probab, k=2):
    Y_pred = np.zeros_like(Y_probab, dtype=int)

    for i in range(Y_probab.shape[0]):
        topk_idx = np.argsort(Y_probab[i])[-k:]
        for idx in topk_idx:
            label = PLUTCHIK_EMOTIONS[idx]
            if Y_probab[i, idx] >= LABEL_THRESHOLDS[label]:
                Y_pred[i, idx] = 1

    return Y_pred


def build_ml_features_for_hybrid(train_text, test_text):
    vectorized = TfidfVectorizer(ngram_range=(1, 2), max_features=60000, min_df=3, max_df=0.9, sublinear_tf=True)

    X_train_tfidf = vectorized.fit_transform(train_text)
    X_test_tfidf = vectorized.transform(test_text)

    nrc = load_lexicon_nrc()
    nrc_emotions_map = build_nrc_emotions_map(nrc)
    nrc_weight = 0.5

    X_train_nrc = extract_nrc_features_process(train_text, nrc_emotions_map)
    X_test_nrc = extract_nrc_features_process(test_text, nrc_emotions_map)

    max_abs = MaxAbsScaler()
    X_train_nrc = max_abs.fit_transform(X_train_nrc)
    X_train_nrc = X_train_nrc * nrc_weight
    X_test_nrc = max_abs.transform(X_test_nrc)
    X_test_nrc = X_test_nrc * nrc_weight

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_nrc)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_nrc)])

    return X_train, X_test, vectorized, max_abs


def vectorize_emotions(emotion):
    vec = np.zeros(len(PLUTCHIK_EMOTIONS), dtype=int)

    if pd.isna(emotion):
        return vec

    labels = emotion.split("|")
    for lbl in labels:
        if lbl in PLUTCHIK_EMOTIONS:
            vec[PLUTCHIK_EMOTIONS.index(lbl)] = 1

    return vec


def build_cnn():
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_WORDS, 256)(inp)
    x = Conv1D(256, 5, activation="relu")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.4)(x)
    out = Dense(len(PLUTCHIK_EMOTIONS), activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


def build_rnn():
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_WORDS, 256)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    attn = tf.keras.layers.Attention()([x, x])

    x = Bidirectional(GRU(128))(attn)
    x = Dropout(0.4)(x)

    out = Dense(len(PLUTCHIK_EMOTIONS), activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


def build_ml_models():
    log = OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight="balanced"))
    linear_svm = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3))
    return log, linear_svm


def evaluate_multilabel_model(Y_true, Y_predictions):
    Y_pred = np.zeros_like(Y_predictions, dtype=int)
    for i, label in enumerate(PLUTCHIK_EMOTIONS):
        Y_pred[:, i] = (Y_predictions[:, i] >= LABEL_THRESHOLDS[label]).astype(int)

    exact_match_acc = np.mean(np.all(Y_true == Y_pred, axis=1))
    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    print("\n---- Evaluation Results ----")
    print(f"Exact-match accuracy : {exact_match_acc:.4f}")
    print(f"Micro F1            : {micro_f1:.4f}")
    print(f"Macro F1            : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(Y_true, Y_pred, target_names=PLUTCHIK_EMOTIONS, zero_division=0))

    print("\nConfusion Matrices (per label):")
    cms = multilabel_confusion_matrix(Y_true, Y_pred)

    for i, label in enumerate(PLUTCHIK_EMOTIONS):
        tn, fp, fn, tp = cms[i].ravel()
        print(f"\n{label.upper()}")
        print(f"TP: {tp} | FP: {fp}")
        print(f"FN: {fn} | TN: {tn}")


if __name__ == "__main__":
    print("\n---- Training Multi-Label Hybrid Ensemble ----")
    emo_df = pd.read_csv(MULTICLASS_EMOTION_LYRIC_PATH)
    emo_df = emo_df.dropna(subset=["lyric", "emotion_NRC"])
    emo_df["emotion_vector"] = emo_df["emotion_NRC"].apply(vectorize_emotions)

    X_text = emo_df["lyric"].astype(str).tolist()
    Y = np.vstack(emo_df["emotion_vector"].values)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    X_pad = pad_sequences(tokenizer.texts_to_sequences(X_text), maxlen=MAX_LEN, padding="post")
    joblib.dump(tokenizer, f"{MULTICLASS_HYBRID_MODEL}/tokenizer.joblib")

    print("\n---- Generating OOF Predictions ----")

    NUM_MODELS = 4
    oof_features = np.zeros((len(X_text), len(PLUTCHIK_EMOTIONS) * NUM_MODELS))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=100)

    for fold, (train, validate) in enumerate(kf.split(X_text)):
        print(f"Fold {fold + 1}/{N_SPLITS}")

        Xtrain_text, Xval_text = np.array(X_text)[train], np.array(X_text)[validate]
        Ytrain = Y[train]

        # ML models
        Xtrain_ml, Xval_ml, tfidf, scaler = build_ml_features_for_hybrid(Xtrain_text, Xval_text)
        lr, svm = build_ml_models()

        lr.fit(Xtrain_ml, Ytrain)
        svm.fit(Xtrain_ml, Ytrain)

        # DL models
        cnn = build_cnn()
        rnn = build_rnn()

        cnn.fit(X_pad[train], Ytrain, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        rnn.fit(X_pad[train], Ytrain, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        oof_features[validate] = np.hstack([
            lr.predict_proba(Xval_ml),
            svm.predict_proba(Xval_ml),
            cnn.predict(X_pad[validate]),
            rnn.predict(X_pad[validate])
        ])

    print("\n---- Training Meta Learner ----")
    meta_model = OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight="balanced"))
    meta_model.fit(oof_features, Y)
    joblib.dump(meta_model, f"{MULTICLASS_HYBRID_MODEL}/meta_learner.joblib")

    print("\n---- Final Models ----")
    X_ml_full, _, tfidf_final, scaler_final = build_ml_features_for_hybrid(X_text, X_text)
    final_lr, final_svm = build_ml_models()

    final_lr.fit(X_ml_full, Y)
    final_svm.fit(X_ml_full, Y)

    joblib.dump(final_lr, os.path.join(MULTICLASS_HYBRID_MODEL, "lr.joblib"))
    joblib.dump(final_svm, os.path.join(MULTICLASS_HYBRID_MODEL, "svm.joblib"))
    joblib.dump(tfidf_final, f"{MULTICLASS_HYBRID_ML_MODEL}/tfidf.joblib")
    joblib.dump(scaler_final, f"{MULTICLASS_HYBRID_ML_MODEL}/nrc_scaler.joblib")

    final_cnn = build_cnn()
    final_rnn = build_rnn()

    final_rnn.fit(X_pad, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    final_cnn.fit(X_pad, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    final_cnn.save(os.path.join(MULTICLASS_HYBRID_DL_MODEL, "cnn.h5"))
    final_rnn.save(os.path.join(MULTICLASS_HYBRID_DL_MODEL, "rnn.h5"))

    print("\n---- Evaluation ----")
    Y_prob = meta_model.predict_proba(oof_features)
    Y_predic = apply_top_k_with_threshold(Y_prob, k=2)
    evaluate_multilabel_model(Y, Y_predic)



