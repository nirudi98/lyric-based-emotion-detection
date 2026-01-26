import pandas as pd
from collections import Counter
import random

from scripts.constants import MOODYLYRICS
from scripts.constants import MOODYLYRICS_4Q

EMOTION_NEIGHBORS = {
    "happy": {"relaxed"},
    "relaxed": {"happy", "sad"},
    "sad": {"relaxed", "angry"},
    "angry": {"sad"},
}

def analyze_dataset(df, name):
    print(f"\nAnalysing dataset : {name}")
    print("Total rows in dataset : ", len(df))

    # check and print duplicated songs in the dataset
    # copy dataset to print duplicates
    print(f"\n---- Check for duplicate songs in {name} ----")
    df = df.copy()
    df["song_key"] = (df["artist"].str.lower().str.strip() + " - " + df["title"].str.lower().str.strip())
    duplicated = df[df.duplicated("song_key", keep=False)]
    if duplicated.empty:
        print("No duplicated songs found.")
    else:
        print(f"Number of duplicated songs found: {duplicated['song_key'].nunique()}")
        for key, group in duplicated.groupby("song_key"):
            artist, song = key.split(" - ", 1)
            print(f"Artist: {artist.title()} | Song: {song.title()} | Count: {len(group)}")

    # removing duplicated songs
    print(f"\n---- Remove duplicate songs in {name} ----")
    original_length = len(df)
    df = df.drop_duplicates(subset=["song_key"]).copy()
    updated_length = len(df)
    print(f"{original_length - updated_length} duplicated songs removed")
    print(f"New row count is {updated_length}")

    print(f"\n---- Emotion-wise song distribution in {name} ----")
    emotion_counts = df["mood"].value_counts()
    print(emotion_counts)

    df.drop(columns=["song_key"], inplace=True)

    return df

def compare_datasets(df1, df2):
    print(f"\nComparing datasets Moody and Moody 4Q....")

    df1 = df1.copy()
    df2 = df2.copy()

    # create unique song identifiers
    df1.loc[:, "song_key"] = (df1["artist"].str.lower().str.strip() + " - " + df1["title"].str.lower().str.strip())
    df2.loc[:, "song_key"] = df2["artist"].str.lower().str.strip() + " - " + df2["title"].str.lower().str.strip()

    # common songs
    common_songs = set(df1["song_key"]).intersection(df2["song_key"])
    print("Number of common songs between datasets : ", len(common_songs))

    if not common_songs:
        print("No common songs found.")

    # identify common songs in both dataset and only retain in one
    # there can be common songs with same emotion or different emotions
    print(f"\n---- Identify common song occurrences ----")
    common_df1 = df1[df1["song_key"].isin(common_songs)]
    common_df2 = df2[df2["song_key"].isin(common_songs)]

    # identify conflicting emotion labels in common songs
    merged = common_df1.merge(common_df2, on="song_key", suffixes=("_df1", "_df2"))
    same_emotion = merged[merged["mood_df1"] == merged["mood_df2"]]
    conflicting_emotion = merged[merged["mood_df1"] != merged["mood_df2"]]

    print("Total number of songs with conflicting emotion labels: ", len(conflicting_emotion))
    print("Total number of songs with same emotion labels: ", len(same_emotion))

    # print common songs with same emotion labels
    if not same_emotion.empty:
        print("\n---- Common songs with same emotion labels ----")
        print(same_emotion[["artist_df1", "title_df1", "mood_df1"]].to_string(index=False))

    if not conflicting_emotion.empty:
        print("\n---- Common songs with conflicting emotion labels ----")
        print(conflicting_emotion[["artist_df1", "title_df1", "mood_df1", "mood_df2"]].to_string(index=False))

    return same_emotion, conflicting_emotion

def resolve_emotion_conflicts(df1, df2, same_emotion_df, conflicting_emotion_df):
    emotion_col = "mood"

    df1 = df1.copy()
    df2 = df2.copy()

    # create unique song identifiers again
    df1.loc[:, "song_key"] = (df1["artist"].str.lower().str.strip() + " - " + df1["title"].str.lower().str.strip())
    df2.loc[:, "song_key"] = df2["artist"].str.lower().str.strip() + " - " + df2["title"].str.lower().str.strip()

    removed_song_keys = []
    remapped_emotion_song_records = []

    # resolve songs with conflicting emotion labels - retain in dataset 2, remove from 1
    # resolve songs with same emotion labels - retain in dataset 2, remove from 1
    print("\n---- Resolve songs with conflicting emotion labels ----")

    # same emotion songs
    same_emotion_song_keys = same_emotion_df["song_key"].unique().tolist()
    removed_song_keys.extend(same_emotion_song_keys)

    # conflicting emotion songs
    global_dist = Counter(list(df1[emotion_col].dropna()) + list(df2[emotion_col].dropna()))

    for _, row in conflicting_emotion_df.iterrows():
        song_key = row["song_key"]
        e1 = row[f"{emotion_col}_df1"]
        e2 = row[f"{emotion_col}_df2"]

        # introduce a balanced resolution
        if e1 == e2:
            chosen = e1
        elif e2 in EMOTION_NEIGHBORS[e1]:
            # when emotions are neighbors random emotion is selected
            chosen = random.choice([e1, e2])
        else:
            # when emotions are not neighbors, global distribution is checked - which emotion is dominant
            chosen = e1 if global_dist[e1] > global_dist[e2] else e2

        remapped_emotion_song_records.append({"artist": row["artist_df1"], "title": row["title_df1"], "dataset1_emotion": e1, "dataset2_emotion": e2,
            "resolved_emotion": chosen
        })

        # update dataset 2 emotion
        df2.loc[df2["song_key"] == song_key, emotion_col] = chosen
        removed_song_keys.append(song_key)

    df1_cleaned = df1[~df1["song_key"].isin(removed_song_keys)].copy()

    print("\n---- Conflict resolution summary ----")
    print(f"Same emotion songs removed from dataset 1: {len(same_emotion_song_keys)}")
    print(f"Conflicting emotion songs resolved: {len(conflicting_emotion_df)}")
    print(f"Total songs removed from dataset 1: {len(set(removed_song_keys))}")

    df1_cleaned.drop(columns=["song_key"], inplace=True)
    df2.drop(columns=["song_key"], inplace=True)

    return df1_cleaned, df2, remapped_emotion_song_records, removed_song_keys

if __name__ == "__main__":
    print("\nLoading Moody datasets...")
    moody = pd.read_csv(MOODYLYRICS)
    moody_4q = pd.read_csv(MOODYLYRICS_4Q)

    # print moody_4q column names
    print("Original Moody column names : ", moody.columns)
    print("Original Moody 4Q column names : ", moody_4q.columns)
    # as 4Q column names are different - change the moody dataset cols - rename cols to align both
    moody.columns = moody.columns.str.lower()
    moody = moody.rename(columns={"emotion": "mood", "song": "title"})
    print("Lowercase Moody column names : ", moody.columns)

    # analyze individually
    moody_clean = analyze_dataset(moody, "Moody Lyrics")
    moody_4q_clean = analyze_dataset(moody_4q, "Moody Lyrics 4Q")
    print("\n----------------------\n")

    # compare datasets to check common songs between
    songs_same_emotion, songs_different_emotions = compare_datasets(moody_clean, moody_4q_clean)

    # there are common songs in both datasets getting duplicated - those should be removed from one dataset
    moody_clean_final, moody_4q_clean_final, remapped_list, removed_list = resolve_emotion_conflicts(moody_clean, moody_4q_clean, songs_same_emotion, songs_different_emotions)
    print("\nRemapped emotion - song list : ", remapped_list)
    print("\nRemoved emotion - song list : ", removed_list)

    print("\n----------------------\n")
    # check dataset sizes and columns again before proceeding
    print(f"Updated Moody column names {moody_clean_final.columns} and dataset size {len(moody_clean_final)}")
    print(f"Updated Moody 4Q column names {moody_4q_clean_final.columns} and dataset size {len(moody_4q_clean_final)}")

    print("\n----------------------\n")
    # now replace with the existing datasets
    print("\nSaving updated Moody datasets...")
    moody_clean_final.to_csv(MOODYLYRICS, index=False)
    moody_4q_clean_final.to_csv(MOODYLYRICS_4Q, index=False)
    print("Moody datasets saved and updated successfully...")
