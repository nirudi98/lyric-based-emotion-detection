import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack, csr_matrix

from scripts.constants import MULTICLASS_DL_ENSEMBLE, MULTICLASS_ML_ENSEMBLE
from scripts.preprocessor.text_preprocessing import load_nrc_lexicon, extract_nrc_features

MAX_WORDS = 50000
MAX_LEN = 300
BATCH_SIZE = 64
EPOCHS = 8
N_SPLITS = 5
THRESHOLD = 0.5

PLUTCHIK_EMOTIONS = [ "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust" ]
LABEL_THRESHOLDS = {
    "anger": 0.5,
    "anticipation": 0.5,
    "disgust": 0.75,
    "fear": 0.5,
    "joy": 0.5,
    "sadness": 0.5,
    "surprise": 0.75,
    "trust": 0.5
}

def predict_emotions_dl_ensemble(text, threshold=THRESHOLD):
    tokenizer = joblib.load(f"{MULTICLASS_DL_ENSEMBLE}/tokenizer.joblib")
    meta = joblib.load(f"{MULTICLASS_DL_ENSEMBLE}/meta_learner.joblib")

    m1 = load_model(f"{MULTICLASS_DL_ENSEMBLE}/bilstm.h5")
    m2 = load_model(f"{MULTICLASS_DL_ENSEMBLE}/cnn.h5")

    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    p1 = m1.predict(pad_seq)
    p2 = m2.predict(pad_seq)
    oof_feat = np.hstack([p1, p2])
    probs = meta.predict_proba(oof_feat)[0]

    top_indices = probs.argsort()[-2:][::-1]
    top_labels = [PLUTCHIK_EMOTIONS[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]

    return list(zip(top_labels, top_probs))

def predict_emotions_ml_ensemble(text):
    ensemble_bundle = joblib.load(f"{MULTICLASS_ML_ENSEMBLE}/ml_ensemble.joblib")
    ml_model = ensemble_bundle["model"]
    tfidf = ensemble_bundle["tfidf"]
    extra_obj = ensemble_bundle["extra_objects"]
    scaler = extra_obj["nrc_scaler"]

    nrc = load_nrc_lexicon()
    X_nrc = np.array(extract_nrc_features([text], nrc))
    X_nrc = scaler.transform(X_nrc)
    X_tfidf = tfidf.transform([text])
    X = hstack([X_tfidf, csr_matrix(X_nrc)])

    Y_prob = ml_model.predict_proba(X)[0]
    # get top 2 emotions
    top_idx = np.argsort(Y_prob)[-2:][::-1]
    top_emotions = [(PLUTCHIK_EMOTIONS[i], float(Y_prob[i])) for i in top_idx]
    return top_emotions
    # Y_pred = np.zeros_like(Y_prob)
    # for i, emo in enumerate(PLUTCHIK_EMOTIONS):
    #     Y_pred[:, i] = (Y_prob[:, i] >= LABEL_THRESHOLDS[emo]).astype(int)
    #
    # predicted_emotions = [
    #     PLUTCHIK_EMOTIONS[i]
    #     for i in range(len(PLUTCHIK_EMOTIONS))
    #     if Y_pred[0, i] == 1
    # ]
    #
    # return predicted_emotions, dict(zip(PLUTCHIK_EMOTIONS, Y_prob[0]))

if __name__ == "__main__":
    # predictions from DL ensemble
    # print("\n---- Multi-Label DL Ensemble Prediction ----")
    # lyric = "surprised"
    # # labels, scores = predict_emotions_dl_ensemble(lyric)
    #
    # print("\nPredicted emotions:", predict_emotions_dl_ensemble(lyric))
    # # print("Scores:", scores)

    # predictions from ML ensemble
    print("\n---- Multi-Label ML Ensemble Prediction ----")
    lyric = "To tell me what to do with my life? Especially when you made a mess. Of every chance you had to success"
    # labels, scores = predict_emotions_dl_ensemble(lyric)

    print("\nPredicted emotions:", predict_emotions_ml_ensemble(lyric))
    # print("Scores:", scores)
