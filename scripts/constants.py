import numpy as np
import pandas as pd
import os

GENRE_MAPPER = {
    'Jazz': 'Jazz', 'Soul': 'Jazz', 'Swing': 'Jazz', 'Gospel': 'Jazz', 'Acid Jazz': 'Jazz',
    'Lounge': 'Jazz',
    'Big Band': 'Jazz', 'Fusion': 'Jazz',

    'Rock': 'Rock', 'Rock & Roll': 'Rock', 'Alternative Rock': 'Rock', 'R&B': 'Rock',
    'Hard Rock': 'Rock',
    'Indie Rock': 'Rock', 'Psychedelic Rock': 'Rock', 'Blues': 'Rock', 'New Wave': 'Rock',
    'Ska': 'Rock', 'Garage Rock': 'Rock',
    'Acoustic': 'Rock', 'Post-Punk': 'Rock', 'Progressive Rock': 'Rock', 'Grunge': 'Rock',
    'HardCore Punk': 'Rock', 'Noise': 'Rock', 'Ska Punk': 'Rock',
    'Pop-Punk': 'Rock', 'Punk Rock': 'Rock', 'Hardcore Punk': 'Rock', 'Classic Rock': 'Rock',
    'Emo': 'Rock', 'Post-Rock': 'Rock', 'Big Beat': 'Rock', 'Goth': 'Rock', 'Stoner Rock': 'Rock',

    'Metal': 'Metal', 'Heavy Metal': 'Metal', 'Thrash Metal': 'Metal', 'Progressive Metal': 'Metal',
    'Symphonic Metal': 'Metal', 'Alternative Metal': 'Metal', 'Doom Metal': 'Metal',
    'Industrial Metal': 'Metal', 'Sludge Metal': 'Metal',
    'Metalcore': 'Metal', 'Rap Metal': 'Metal', 'Post-Hardcore': 'Metal', 'Death Metal': 'Metal',
    'Grindcore': 'Metal', 'Gothic Metal': 'Metal', 'Black Metal': 'Metal', 'Nu Metal': 'Metal',
    'Speed Metal': 'Metal',
    'Hardcore': 'Metal', 'Folk Metal': 'Metal', 'Glam Metal': 'Metal',

    'BlueGrass': 'Country', 'Country': 'Country', 'Folk': 'Country',
    'Alternative Country': 'Country', 'Country Rock': 'Country',

    'Pop': 'Pop', 'Synthpop': 'Pop', 'Country Pop': 'Pop', 'Indie Pop': 'Pop', 'Indie': 'Pop',
    'Pop-Rock': 'Pop', 'J-Pop': 'Pop', 'SynthPop': 'Pop',
    'Reggae': 'Pop', 'Latin': 'Pop', 'Funk': 'Pop', 'Dance': 'Pop', 'Drum & Bass': 'Pop',
    'Euro Dance': 'Pop',
    'Disco': 'Pop',

    'Hip-Hop': 'Hip-Hop', 'Rap': 'Hip-Hop', 'Alternative Hip-Hop': 'Hip-Hop', 'Grime': 'Hip-Hop',
    'Trip Hop': 'Hip-Hop',

    'Electronic': 'Electronic', 'Deep House': 'Electronic', 'House': 'Electronic',
    'Techno': 'Electronic',
    'Electro House': 'Electronic', 'Trance': 'Electronic', 'Breaks': 'Electronic',
    'New Age': 'Electronic', 'Dubstep': 'Electronic',
    'Ambient': 'Electronic', 'Electro-Industrial': 'Electronic', 'UK Garage': 'Electronic',

    'World/Ethnic': None, 'Avant-Garde': None, 'Comedy': None, 'Downtempo': None, 'Classical': None,
    'Experimental': None, 'Singer Songwriter': None,
     None: None, pd.NA: None, np.nan: None, '...': None,
}

TEXT_WITH_CHORUS_VERSE_IN_BRACKETS = r'[\(\[].*?[\)\]]'
VA_LIMIT = 0.25
LIMIT = 10000
LIMIT_WITHOUT_BRACKETS = 5500

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# unprocessed datasets - to download lyrics
DATASET_PATH = os.path.join(PROJECT_PATH, 'datasets/unprocessed_datasets')
PREPROCESSED_DATASET_PATH = os.path.join(PROJECT_PATH, 'datasets/processed_datasets')
LEXICON_PATH = os.path.join(PROJECT_PATH, 'lexicons')
MULTICLASS_DATASET_PATH = os.path.join(PROJECT_PATH, 'datasets/multiclass_datasets')

# filtered datasets, processed and not processed
SONG_DATASET_PATH = os.path.join(DATASET_PATH, "filtered_dataset.csv")
PREPROCESSED_LYRIC_PATH = os.path.join(PREPROCESSED_DATASET_PATH, "preprocessed_dataset.csv")
PREPROCESSED_EMOTION_LYRIC_PATH = os.path.join(PREPROCESSED_DATASET_PATH, "lyrics_with_emotion_dataset.csv")

# filtered datasets, processed and not processed - multiclass classification
MULTICLASS_EMOTION_LYRIC_PATH = os.path.join(MULTICLASS_DATASET_PATH, "lyrics_with_emotion_dataset.csv")


INPUT_FILE = os.path.join(DATASET_PATH, 'merged_datasets_V1.csv')
OUTPUT_FILE = os.path.join(DATASET_PATH, 'lyrics.csv')

# lyric csv file paths
MOODY_FOLDER = os.path.join(PROJECT_PATH, "datasets/Moody")
MOODYLYRICS = os.path.join(MOODY_FOLDER, "MoodyLyrics/MoodyLyrics.csv")
MOODYLYRICS_4Q = os.path.join(MOODY_FOLDER, "MoodyLyrics4Q/MoodyLyrics4Q.csv")
MOODY_MERGED = os.path.join(MOODY_FOLDER, "MoodyMerged/MoodyMerged.csv")
MOODY_LYRICS = os.path.join(MOODY_FOLDER, "MoodyMerged/MoodyMerged_lyrics.csv")
MOODY_LYRICS_TEST = os.path.join(MOODY_FOLDER, "MoodyMerged/MoodyMerged_lyrics_test.csv")

# API tokens
AUDIO_DB_URL = "https://www.theaudiodb.com/api/v1/json/2/searchtrack.php"

# error logs when fetching song lyrics
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LYRICS_FETCHED_PATH = os.path.join(PROJECT_PATH, "datasets/helper_logs")
LYRICS_FETCHED = os.path.join(LYRICS_FETCHED_PATH, "lyrics_fetch_log.txt")
CLEANED_LYRICS_FETCHED = os.path.join(LYRICS_FETCHED_PATH, "cleaned_lyrics_fetch_log.txt")
REPROCESS_LYRICS = os.path.join(LYRICS_FETCHED_PATH, "reprocessed_lyrics.txt")

# model save paths
MODEL_PATH = os.path.join(PROJECT_PATH, "models/comparative_models")
TEST_MODEL_SAVE_LOAD_PATH = os.path.join(PROJECT_PATH, "models/save_load_models")
DL_MODEL_PATH = os.path.join(PROJECT_PATH, "models/comparative_models/dl_models")

# multi-label classification model paths
MULTICLASS_MODEL_PATH = os.path.join(PROJECT_PATH, "models/comparative_models/multilabel_models")
MULTICLASS_DL_ENSEMBLE = os.path.join(MULTICLASS_MODEL_PATH, "dl_ensemble")
MULTICLASS_ML_ENSEMBLE = os.path.join(MULTICLASS_MODEL_PATH, "ml_ensemble")
MULTICLASS_BASE_ML_MODELS = os.path.join(MULTICLASS_MODEL_PATH, "ml_ensemble/base_models")
MULTICLASS_HYBRID_MODEL = os.path.join(MULTICLASS_MODEL_PATH, "hybrid_ensemble")
MULTICLASS_HYBRID_ML_MODEL = os.path.join(MULTICLASS_MODEL_PATH, "hybrid_ensemble/ml_models")
MULTICLASS_HYBRID_DL_MODEL = os.path.join(MULTICLASS_MODEL_PATH, "hybrid_ensemble/dl_models")




