import pandas as pd
from collections import Counter
import re

LEXICON_FILEPATH = "/Users/niruddya/Documents/IIT/second_year/research/lit review papers/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# loading NRC lexicon
def load_nrc_lexicon():
    nrc = pd.read_csv(LEXICON_FILEPATH, sep="\t", names=["word", "emotion", "association"])

    # 8 emotions - anger/ fear/ anticipation/ trust/ surprise/ sadness/ joy/ disgust
    # sentiment polarity categories - positive/ negative
    nrc = nrc[nrc["association"] == 1].drop(columns="association")
    emotion_labels = {"anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"}
    sentiment_labels = {"positive", "negative"}

    nrc["type"] = nrc["emotion"].apply(lambda x: "sentiment" if x in sentiment_labels else "emotion")
    nrc["sentiment"] = nrc["emotion"].apply(lambda x: x if x in sentiment_labels else None)
    nrc["emotion_label"] = nrc["emotion"].apply(lambda x: x if x in emotion_labels else None)

    print(nrc.head(5))
    return nrc

# analyzing text and categorizing
def analyze_emotions_and_sentiment(nrc):
    words = re.findall(r"\b\w+\b", text.lower())
    matched = nrc[nrc["word"].isin(words)]

    emotions = Counter(matched["emotion"].dropna())
    sentiments = Counter(matched["sentiment"].dropna())

    print({"emotions": dict(emotions),"sentiment": dict(sentiments)})

    # normalizing
    total_emotions = sum(emotions.values())
    emotions_normalized = {emotion: round(count/ total_emotions, 2) for emotion, count in emotions.items()} if total_emotions > 0 else {}
    total_sentiments = sum(sentiments.values())
    sentiments_normalized = {sentiment: round(count/ total_sentiments, 2) for sentiment, count in sentiments.items()} if total_sentiments > 0 else {}
    return emotions_normalized, sentiments_normalized

def categorize_sentence(analysis_result):
    emotions, sentiments = analysis_result

    if emotions:
        dominant_emotion = max(emotions, key=emotions.get)
        dominant_emotion_scor = emotions[dominant_emotion]
    else:
        dominant_emotion, dominant_emotion_scor = None, None

    if sentiments:
        dominant_sentiment = max(sentiments, key=sentiments.get)
        dominant_sentiment_scor = sentiments[dominant_sentiment]
    else:
        dominant_sentiment, dominant_sentiment_scor = None, None

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_score": dominant_emotion_scor,
        "dominant_sentiment": dominant_sentiment,
        "dominant_sentiment_score": dominant_sentiment_scor
    }

if __name__ == '__main__':
    nrc_lexicon = load_nrc_lexicon()
    text = "I trust people"
    result = analyze_emotions_and_sentiment(nrc_lexicon)
    final_emotion = categorize_sentence(result)
    print(final_emotion)