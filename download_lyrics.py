import re
import numpy as np

import lyricsgenius
import pandas as pd
import requests
from langdetect import detect
from requests.exceptions import HTTPError
from urllib.parse import quote_plus

from scripts.constants import MOODYLYRICS, MOODYLYRICS_4Q, MOODY_MERGED, AUDIO_DB_URL, REPROCESS_LYRICS, MOODY_LYRICS, \
    SONG_DATASET_PATH
from scripts.constants import CLEANED_LYRICS_FETCHED, GENRE_MAPPER

COLS_REQUIRED = ['song_id', 'dataset', 'title', 'artist', 'genre', 'arousal_mean', 'arousal_std', 'valence_mean', 'valence_std', 'emotion_4Q', 'emotion_2Q']

# client id = 1pgGdomlex7LkO4WtqzBnDtG4GeHt180EqkENyVmjZ20xDoUUQgXiHyoO4VQcDe1
# client secret = AsHKqv3oi9_iURWs7N1rDbVns3jGEe5Jbg8b5erUpYuThDmHblpcN0EdWsePefJu4Bi8V8nHzZTGhRDDagdOlw
# access token = mgkGdpCCLySd69_gUgzx2cf6uvxunaVYsPbKrr77qowS8PF6Q-_35qgrYUQVHWTo

GENIUS_API_ACCESS_TOKEN = "mgkGdpCCLySd69_gUgzx2cf6uvxunaVYsPbKrr77qowS8PF6Q-_35qgrYUQVHWTo"
genius = lyricsgenius.Genius(GENIUS_API_ACCESS_TOKEN, verbose=False)

# standardize the existing datasets
def standardize_datasets():
    print(f"\n---- Standardize Moody Datasets ----")
    moodylyrics = pd.read_csv(MOODYLYRICS)
    moodylyrics_4q = pd.read_csv(MOODYLYRICS_4Q)

    for col in ['genre', 'arousal_mean', 'arousal_std', 'valence_mean', 'valence_std']:
        moodylyrics[col] = np.nan
        moodylyrics_4q[col] = np.nan

    moodylyrics = moodylyrics.rename(columns={'index': 'song_id', 'mood': 'emotion_4Q'})
    moodylyrics_4q = moodylyrics_4q.rename(columns={'index': 'song_id', 'mood': 'emotion_4Q'})
    moodylyrics['emotion_2Q'] = moodylyrics['emotion_4Q'].apply(set_sentiments)
    moodylyrics_4q['emotion_2Q'] = moodylyrics_4q['emotion_4Q'].apply(set_sentiments)

    moodylyrics['dataset'] = 'MoodyLyrics'
    moodylyrics_4q['dataset'] = 'MoodyLyrics4Q'
    moodylyrics = moodylyrics[COLS_REQUIRED]
    moodylyrics_4q = moodylyrics_4q[COLS_REQUIRED]

    # merge both datasets
    merged_df = pd.concat([moodylyrics, moodylyrics_4q], ignore_index=True)
    merged_df.to_csv(MOODY_MERGED, index=False)
    print(f"Saved merged dataset")
    print(f"Total rows in merged Moody dataset: {len(merged_df)}")

    return merged_df

# create 2 sentiments - if needed
def set_sentiments(emotion):
    if emotion == 'happy' or emotion == 'relaxed':
        return 'positive'
    else:
        return 'negative'

# fetching genre of the songs
def get_genre(title, artist):
    print(f"\n---- Genre details fetched from AudioDB API ----")
    base_url = AUDIO_DB_URL
    url = f"{base_url}?s={quote_plus(artist)}&t={quote_plus(title)}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get('track'):
            genre = data['track'][0].get('strGenre')
            print(f"Genre fetched: {genre}")
            return genre
        else:
            print("No track found in response.")
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

# fetching lyrics for each song in df
def get_song_lyrics(title, artist):
    print(f"---- Fetching song lyrics for {title} ----")
    try:
        song = genius.search_song(title, artist)
        if song and song.artist != "Spotify":
            return re.sub(r"\n", " ", song.lyrics)
    except Exception as e:
        print(f"Lyrics fetch failed for {title} - {artist}: {e}")
    return None

def get_language(lyrics):
    return detect(lyrics) if lyrics else None

# try to fetch the failed lyric fetch
def reprocess_failed_lyrics():
    print(f"\n---- Reprocessing failed lyrics ----")
    with open(CLEANED_LYRICS_FETCHED, "r", encoding="utf-8") as infile, \
         open(REPROCESS_LYRICS, "w", encoding="utf-8") as outfile:

        outfile.write("song|artist|lyrics\n")
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                title, artist = [x.strip() for x in line.split("|", 1)]
                print(f"Processing {title} - {artist}")
            except ValueError:
                continue
            lyrics = get_song_lyrics(title, artist)

            lyrics = lyrics if lyrics else ""
            lyrics = lyrics.replace("|", " ")

            outfile.write(f"{title}|{artist}|{lyrics}\n")

    print(f"\n---- Reprocessing failed genre ----")
    genre_fetch = pd.read_csv(REPROCESS_LYRICS, sep="|", names=["song", "artist", "lyrics"], engine="python")
    genre_fetch["genre"] = None

    # fetch the missing genre
    genre_fetch["genre"] = genre_fetch.apply(lambda row: get_genre(row["song"], row["artist"]), axis=1)
    genre_fetch.to_csv(REPROCESS_LYRICS, sep="|", index=False, header=False)

    print(f"\n---- Set corresponding emotions ----")
    # fetch the mood mapped in the original MoodyLyrics4Q file of the songs to the txt
    original_moody = pd.read_csv(MOODY_MERGED)
    txt_moody = pd.read_csv(REPROCESS_LYRICS, sep="|", names=["song", "artist", "lyrics", "genre"], header=None)

    # Normalize for safe mapping (lowercase, strip spaces)
    original_moody["title_key"] = original_moody["title"].str.lower().str.strip()
    original_moody["artist_key"] = original_moody["artist"].str.lower().str.strip()

    txt_moody["title_key"] = txt_moody["song"].str.lower().str.strip()
    txt_moody["artist_key"] = txt_moody["artist"].str.lower().str.strip()

    txt_moody = txt_moody.merge(original_moody[["title_key", "artist_key", "emotion_4Q", "emotion_2Q"]],
                                on=["title_key", "artist_key"], how="left")
    txt_moody = txt_moody.drop(columns=["title_key", "artist_key"])
    txt_moody = txt_moody[["song", "artist", "lyrics", "genre", "emotion_4Q", "emotion_2Q"]]

    txt_moody.to_csv(REPROCESS_LYRICS, sep="|", index=False, header=False, encoding="utf-8")
    print(f"Songs file updated with emotion")

    print(f"\n---- Merge the completed songs to original lyrics ----\n")
    completed_list = pd.read_csv(MOODY_LYRICS)
    completed_list_copy = completed_list.copy()
    print(f"Original dataset length before merging: {len(completed_list_copy)}\n")

    completed_txt_list = pd.read_csv(REPROCESS_LYRICS, sep="|",
                                     names=["song", "artist", "lyrics", "genre", "emotion_4Q", "emotion_2Q"],
                                     header=None,
                                     encoding="utf-8",
                                     skip_blank_lines=True)
    completed_txt_list = completed_txt_list.rename(columns={"song": "title"})
    first_row = completed_txt_list.iloc[0]
    if first_row["title"].lower() == "title" or first_row["artist"].lower() == "artist":
        completed_txt_list = completed_txt_list.iloc[1:].reset_index(drop=True)

    # song ids should be created for the songs
    if completed_list["song_id"].notna().any():
        last_id = completed_list["song_id"].iloc[-1]
        match = re.match(r"([A-Za-z]+)(\d+)", str(last_id))
        if match:
            prefix, num = match.groups()
            num = int(num)
        else:
            prefix = "ML"
            num = len(completed_list)
    else:
        prefix = "ML"
        num = 0

    completed_txt_list["song_id"] = [f"{prefix}{num + i + 1}" for i in range(len(completed_txt_list))]
    completed_txt_list["dataset"] = "MoodyLyrics4Q"  # dataset name
    completed_txt_list["arousal_mean"] = None
    completed_txt_list["arousal_std"] = None
    completed_txt_list["valence_mean"] = None
    completed_txt_list["valence_std"] = None
    completed_txt_list["language"] = "en"

    completed_txt_list = completed_txt_list[["song_id", "dataset", "title", "artist", "genre",
                     "arousal_mean", "arousal_std", "valence_mean", "valence_std",
                     "emotion_4Q", "emotion_2Q", "lyrics", "language"]]

    completed_list_copy = pd.concat([completed_list, completed_txt_list], ignore_index=True)
    completed_list_copy.to_csv(MOODY_LYRICS, index=False, encoding="utf-8")

    print(f"Songs text file is added to original dataset, new dataset length: {len(completed_list_copy)}")

def filter_dataset():
    print(f"---- Filtering dataset ----")
    before_filter = pd.read_csv(MOODY_LYRICS)
    print("List of columns available before processing ", before_filter.columns.tolist())

    # check number of duplicates with lyrics
    duplicates_mask = before_filter['lyrics'].duplicated(keep=False)
    duplicated_songs = before_filter[duplicates_mask][['song_id', 'title', 'lyrics']]

    # print duplicated songs
    print("Duplicated songs based on lyrics:")
    for idx, row in duplicated_songs.iterrows():
        print(f"Song ID: {row['song_id']} | Title: {row['title']}")
        print(f"Lyrics: {row['lyrics'][:100]}...")
        print("-" * 60)

    # drop the duplicates based on the 'lyrics' column, keeping the first occurrence
    print(f"Old dataset shape before removing duplicates: {before_filter.shape}")
    before_filter = before_filter.drop_duplicates(subset=['lyrics'], keep='first').reset_index(drop=True)

    print(f"New dataset shape after removing duplicates: {before_filter.shape}")

    # check number of duplicates with title and artist
    # duplicates_title_artist = before_filter[before_filter.duplicated(subset=['title', 'artist'], keep=False)]
    # print("Exact duplicated songs (title + artist):")
    # print(duplicates_title_artist[['song_id', 'title', 'artist']])
    # this is 0 so no issue
    print(f"---- Check genres in dataset ----")
    before_filter['updated_genre'] = before_filter['genre'].map(GENRE_MAPPER)
    unknown_genres = set(before_filter['genre'].unique()) - set(GENRE_MAPPER.keys())
    print("Genres not in mapper:", unknown_genres)

    before_filter['genre'] = before_filter['updated_genre']
    before_filter.drop(columns=['updated_genre'], inplace=True)
    print("List of columns available after processing ", before_filter.columns.tolist())
    before_filter.to_csv(SONG_DATASET_PATH, index=False)

if __name__ == '__main__':
    song_df = standardize_datasets()
    print(f"Number of rows in merged song dataset: {len(song_df)}")
    print(f"Columns in merged song dataset: {len(song_df.columns)}")
    # fetch lyrics for the songs
    print(f"\n---- Fetch lyrics for Moody datasets ----")
    song_df["lyrics"] = song_df.apply(lambda r: r["lyrics"] if pd.notna(r.get("lyrics")) else get_song_lyrics(r["title"], r["artist"]),axis=1,)

    # this project focuses on English lang only
    print(f"\n---- Filtering languages ----")
    song_df["language"] = song_df["lyrics"].apply(lambda lyrics: detect(lyrics) if lyrics else None)
    df = song_df[song_df["language"] == "en"]

    # fetch genres of songs
    print(f"\n---- Fetch genres for songs ----")
    genre_missed = df["genre"].isna() | (df["genre"] == "null")
    df.loc[genre_missed, "genre"] = df.loc[genre_missed].apply(lambda r: get_genre(r["title"], r["artist"]), axis=1)

    # save the lyrically enriched dataset
    df.to_csv(MOODY_LYRICS, index=False)

    # processing the songs that failed to fetch lyrics
    # reprocess_failed_lyrics()
    # reprocessing is done comment out the code

    # filter and recheck the final dataset
    # filter_dataset()



