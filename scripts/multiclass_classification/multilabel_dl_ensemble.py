import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

import tensorflow as tf
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense,
    Dropout, Conv1D, GlobalMaxPooling1D, GRU
)
from transformers import DistilBertTokenizerFast, TFDistilBertModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scripts.constants import MULTICLASS_EMOTION_LYRIC_PATH, MULTICLASS_DL_ENSEMBLE

MAX_WORDS = 50000
MAX_LEN = 300
BATCH_SIZE = 64
EPOCHS = 8
N_SPLITS = 5
THRESHOLD = 0.5

PLUTCHIK_EMOTIONS = [ "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust" ]
LABEL_THRESHOLDS = {"anger": 0.5, "anticipation": 0.5, "disgust": 0.75,  "fear": 0.5, "joy": 0.5, "sadness": 0.5, "surprise": 0.75, "trust": 0.5}


def vectorize_emotions(emotion):
    vec = np.zeros(len(PLUTCHIK_EMOTIONS), dtype=int)

    if pd.isna(emotion):
        return vec

    labels = emotion.split("|")
    for lbl in labels:
        if lbl in PLUTCHIK_EMOTIONS:
            vec[PLUTCHIK_EMOTIONS.index(lbl)] = 1

    return vec


def build_bilstm():
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_WORDS, 256)(inp)
    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.4)(x)
    out = Dense(len(PLUTCHIK_EMOTIONS), activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


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


def evaluate_multilabel_model(Y_true, predicted_y):
    y_pred_bin = np.zeros_like(predicted_y, dtype=int)

    # applying label specific thresholds
    for i, label in enumerate(PLUTCHIK_EMOTIONS):
        y_pred_bin[:, i] = (predicted_y[:, i] >= LABEL_THRESHOLDS[label]).astype(int)

    # Exact-match accuracy
    exact_match = np.all(Y_true == y_pred_bin, axis=1).mean()
    print(f"\nExact-match accuracy: {exact_match:.4f}")

    # Per-class accuracy
    per_class_acc = (Y_true == y_pred_bin).sum(axis=0) / Y_true.shape[0]
    for idx, label in enumerate(PLUTCHIK_EMOTIONS):
        print(f"Accuracy for {label}: {per_class_acc[idx]:.4f}")

    macro_f1 = f1_score(Y_true, y_pred_bin, average='macro')
    micro_f1 = f1_score(Y_true, y_pred_bin, average='micro')
    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(Y_true, y_pred_bin, target_names=PLUTCHIK_EMOTIONS))

    return {"y_pred": y_pred_bin, "micro_f1": micro_f1, "macro_f1": macro_f1}


# this is a function written to evaluate each DL model - to choose for the ML+DL model
def evaluate_best_dl_models(X_inputs, Y_true, model_list):
    print("\n---- Evaluating Individual DL Models ----")
    results = []

    for name, model in model_list.items():
        print(f"\n--- Evaluating {name} ---")
        Y_prob = model.predict(X_inputs)
        eval_metrics = evaluate_multilabel_model(Y_true, Y_prob)
        results.append({ "model_name": name, "micro_f1": eval_metrics["micro_f1"], "macro_f1": eval_metrics["macro_f1"],
            "y_pred": eval_metrics["y_pred"]
        })

    results_sorted = sorted(results, key=lambda x: x["micro_f1"], reverse=True)
    print("\n--- Model Ranking by Micro F1 ---")
    for r in results_sorted:
        print(f"{r['model_name']}: Micro F1 = {r['micro_f1']:.4f}, Macro F1 = {r['macro_f1']:.4f}")

    return results_sorted


if __name__ == "__main__":
    print("\n---- Training Multi-Label DL Ensemble ----")
    emo_df = pd.read_csv(MULTICLASS_EMOTION_LYRIC_PATH)
    emo_df = emo_df.dropna(subset=["lyric", "emotion_NRC"])
    emo_df["emotion_vector"] = emo_df["emotion_NRC"].apply(vectorize_emotions)

    X_text = emo_df["lyric"].astype(str).tolist()
    Y = np.vstack(emo_df["emotion_vector"].values)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    X_seq = tokenizer.texts_to_sequences(X_text)
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")
    joblib.dump(tokenizer, os.path.join(MULTICLASS_DL_ENSEMBLE, "tokenizer.joblib"))

    print("\n---- Generating OOF Predictions ----")

    NUM_MODELS = 3
    oof_features = np.zeros((X_pad.shape[0], len(PLUTCHIK_EMOTIONS) * NUM_MODELS))
    strat_labels = np.argmax(Y, axis=1)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=100)

    for fold, (tr, va) in enumerate(skf.split(X_pad, strat_labels)):
        print(f"Fold {fold + 1}/{N_SPLITS}")

        m1 = build_bilstm()
        m2 = build_cnn()
        m3 = build_rnn()

        m1.fit(X_pad[tr], Y[tr], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        m2.fit(X_pad[tr], Y[tr], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        m3.fit(X_pad[tr], Y[tr], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

        p1 = m1.predict(X_pad[va])
        p2 = m2.predict(X_pad[va])
        p3 = m3.predict(X_pad[va])

        print("p1:", p1.shape, "p2:", p2.shape, "p3:", p3.shape)

        oof_features[va] = np.hstack([p1, p2, p3])

    print("\n---- Training Meta Learner ----")
    meta_model = OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight="balanced"))
    meta_model.fit(oof_features, Y)

    joblib.dump(meta_model, f"{MULTICLASS_DL_ENSEMBLE}/meta_learner.joblib")
    final_bilstm = build_bilstm()
    final_cnn = build_cnn()
    final_rnn = build_rnn()

    final_bilstm.fit(X_pad, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    final_cnn.fit(X_pad, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    final_rnn.fit(X_pad, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    final_bilstm.save(f"{MULTICLASS_DL_ENSEMBLE}/bilstm.h5")
    final_cnn.save(f"{MULTICLASS_DL_ENSEMBLE}/cnn.h5")
    final_rnn.save(f"{MULTICLASS_DL_ENSEMBLE}/rnn.h5")

    # evaluating individual models
    models_dict = { "BiLSTM": final_bilstm, "CNN": final_cnn, "RNN": final_rnn }
    best_models = evaluate_best_dl_models(X_pad, Y, models_dict)

    print("\n---- Evaluation ----")
    p1 = final_bilstm.predict(X_pad)
    p2 = final_cnn.predict(X_pad)
    p3 = final_rnn.predict(X_pad)
    print("\nDuring evaluation p1:", p1.shape, "p2:", p2.shape, "p3:", p3.shape)
    oof_features_final = np.hstack([p1, p2, p3])
    Y_pred = meta_model.predict_proba(oof_features_final)

    evaluate_multilabel_model(Y, Y_pred)



