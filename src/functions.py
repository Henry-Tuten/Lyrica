import pandas as pd
import re
def filter_csv_columns(original_csv_path, new_csv_path, selected_columns):
    # Load the original CSV file
    df = pd.read_csv(original_csv_path)

    # Select the columns
    new_df = df[selected_columns]

    # Write the new DataFrame to a CSV file
    new_df.to_csv(new_csv_path, index=False)

def get_num_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def count_matching_entries(top_hits_csv_path, song_lyrics_csv_path):
    # Load the CSV files into pandas DataFrames
    top_hits_df = pd.read_csv(top_hits_csv_path)
    song_lyrics_df = pd.read_csv(song_lyrics_csv_path)

    # Create sets of unique song names from both DataFrames
    top_hits_songs = set(top_hits_df['Song Name'].unique())
    song_lyrics_songs = set(song_lyrics_df['title'].unique())

    # Find the intersection of the two sets (i.e., songs present in both CSVs)
    matching_songs = top_hits_songs & song_lyrics_songs

    # Return the number of matching songs
    return len(matching_songs)

def create_lyrics_csv(top_hits_csv_path, song_lyrics_csv_path, new_csv_path):
    # Load the CSV files into pandas DataFrames
    top_hits_df = pd.read_csv(top_hits_csv_path)
    song_lyrics_df = pd.read_csv(song_lyrics_csv_path)

    # Get unique song names from both DataFrames
    top_hits_songs = set(top_hits_df['Song Name'].unique())
    song_lyrics_songs = set(song_lyrics_df['title'].unique())

    # Find the intersection of the two sets (i.e., songs present in both CSVs)
    matching_songs = top_hits_songs & song_lyrics_songs

    # Filter song_lyrics_df to only include matching songs
    matching_lyrics_df = song_lyrics_df[song_lyrics_df['title'].isin(matching_songs)][['title', 'lyrics']]

    # Write the matching song titles and lyrics to a new CSV file
    matching_lyrics_df.to_csv(new_csv_path, index=False)

def clean_lyrics(lyrics):
    # Remove brackets and their contents
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    # Replace newline characters with spaces
    lyrics = re.sub(r'\n', ' ', lyrics)

    # Remove special characters (except spaces)
    lyrics = re.sub(r'[^\w\s]', '', lyrics)

    # Replace multiple spaces with a single space
    lyrics = re.sub(r'\s+', ' ', lyrics)

    return lyrics.strip()

def clean_lyrics_in_csv(input_csv_path, output_csv_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # Apply the clean_lyrics function to every row of the 'lyrics' column
    df['lyrics'] = df['lyrics'].apply(clean_lyrics)

    # Write the DataFrame with the cleaned lyrics to a new CSV file
    df.to_csv(output_csv_path, index=False)