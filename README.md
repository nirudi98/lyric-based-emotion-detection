Scripts to Run in Order

1. validate_datasets.py : This script checks both datasets used Moody Lyrics and Moody Lyrics 4Q for inconsistencies between them before downloading lyrics.
2. download_lyrics.py : This script download lyrics and genres. The song lyrics that weren't downloaded in the first go are filtered to a temporary text file to reprocess. After everything is completed, the remaining set is also attached to the merged dataset; Moody Lyrics Merged.
3. nrc_emotion_extension.py : This script extends the existing 4Q emotion space to 8Q emotion space, only keeping the necessary columns
4. text_preprocessing.py : This script contains all text preprocessing functions. This gets executed when extending emotions to 8Q
5. visualize_dataset.py : This script has functions to analyze the final dataset ready for the model training

Datasets
1. Moody Merged
2. filtered_dataset : filtered dataset after lyrics are downloaded and duplicates removed
3. preprocessed_dataset : duplicates removed and text preprocessing is done to song lyrics, null lyrics entries removed
4. lyrics_with_emotion_dataset : all songs with 4Q emotions are now mapped to 8Q emotions, sentiments are added for positive/negative along with the confidence score count