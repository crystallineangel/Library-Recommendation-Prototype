# download more data (just let it run)
# HTML, CSS, Flask
# can try to get fb

import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
from streamlit import title

# streamlit needs the prototype/combined thing
df = pd.read_csv("complete_data.csv")

"""TOP TEN"""
# given by ChatGPT
def get_book_list(df):
    """Return a formatted list of books as 'Title -- Author'."""
    # .apply lambda etc: formats the list so that it displays accordingly and is converted into a list
    return df.apply(lambda row: f"{row['title']} -- {row['author']}", axis=1).tolist()

def unique_genres(df):
    """Returns a list of unique genres"""

    def safe_eval(value):
        try:
            # only evaluates lists, not invalid strings like "no genres found"
            # ast idea by ChatGPT
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                return ast.literal_eval(value)
            return []
        except:
            return []

    df['genre'] = df['genre'].apply(safe_eval)

    # ChatGPT's code, essentially writes out all individual genres from the sub-lists,
    # then turns it into a sorted set (i.e., no duplicates)
    all_genres = [genre for sublist in df['genre'] for genre in sublist if genre]
    unique_genres = sorted(set(all_genres))

    return unique_genres

# Done by ChatGPT
def filter_by_genre(df, selected_genres):
    """Filters books that match at least one of the selected genres."""
    return df[df['genre'].apply(lambda genres: any(g in selected_genres for g in genres))]

# Done by ChatGPT
# todo: i should rename this function
def get_top_books_by_genre(df, selected_genres):
    """Returns the top 10 books for each selected genre."""
    filtered_df = filter_by_genre(df, selected_genres)
    top_books = {}
    for genre in selected_genres:
        genre_books = filtered_df[filtered_df['genre'].apply(lambda g: genre in g)]
        top_books[genre] = genre_books.nlargest(10, 'rating')

    return top_books


"""NEAREST NEIGHBORS"""

# cleaning code (1) given by ChatGPT
# todo: go through and comment so i understand this
def clean_genres_1(value):
    try:
        # Check if the value is the string "No genres found" and return an empty list
        if isinstance(value, str) and value.lower() == "no genres found":
            return []  # No genres found, so return an empty list
        # Only evaluate values that look like lists
        elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            return ast.literal_eval(value)  # Convert string representation of a list into an actual list
        elif isinstance(value, str):
            return [value]  # If it's a string but not a list-like string, turn it into a single-item list
        return value  # Return the value if it's already a list
    except Exception as e:
        print(f"Error evaluating genre: {e}")
        return []

def clean_genres_2(df):
    """Ensure that genre column contains lists of genres, and empty genres are handled as empty lists"""
    df['genre'] = df['genre'].apply(clean_genres_1)
    return df

def genre_binary(df):
    """Transforms genres to binary"""
    # todo: go through why it didn't work previously
    binarizer = MultiLabelBinarizer()
    genre_matrix = binarizer.fit_transform(df['genre'])
    df_genres = pd.DataFrame(genre_matrix, columns=binarizer.classes_)
    return df_genres

def clean_descr(df):
    """Ensure that missing descriptions are handled by replacing them with empty strings"""
    df['descr'] = df['descr'].fillna('')
    return df

def descr_numerical(df):
    """Makes descriptions numerical"""
    # Tfid was given by ChatGPT, aka Term Frequency-Inverse Document Frequency which turns words into numbers
    # Stop_words stop the most common English words
    # Max features ensures there aren't too many unique values, which wouldn't be useful
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    descr_matrix = vectorizer.fit_transform(df['descr'])
    return descr_matrix

def get_recs(df, title):
    """Gets recommendations based on nearest features, including genre, description, page length, and rating"""

    # cleans everything
    df = clean_genres_2(df)
    df = clean_descr(df)

    print(df.head())

    # ensures pages and ratings are numerical
    # df['pages'] = pd.to_numeric(df['pages'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # standardizes everything so the page difference doesn't mess up the cosine rec system/algorithm
    # todo: I'll do it later once i get the main thing to work, currently an object and not a float
    # numerical_features = scale.fit_transform(df[['rating']].values)

    # simplifies stuff and calls functions
    df_genres = genre_binary(df)
    descr_matrix = descr_numerical(df)

    """Features matrix to be used with NN"""
    feature_matrix = hstack([df_genres, descr_matrix, np.array(df['rating']).reshape(-1, 1)])

    # converts the matrix into a dense one so that things work out
    # solution given by ChatGPT
    feature_matrix_dense = feature_matrix.toarray()

    # sets up the nearest neighbors algorithm
    knn = NearestNeighbors(n_neighbors=10, metric = 'cityblock')  # closest 10
    knn.fit(feature_matrix_dense) # trains

    book_index = df[df['title'] == title].index[0]

    distances, indices = knn.kneighbors(feature_matrix_dense[book_index].reshape(1, -1), n_neighbors=10)

    recommended_books = df.iloc[indices[0][1:]]  # Exclude the first result (same book)

    return recommended_books[['title', 'author']]

def get_rating(df, title):
    matching_row = df[df['title'].str.lower() == title.lower()]

    if not matching_row.empty:
        return matching_row['rating'].values[0]

def get_descr(df, title):
    matching_row = df[df['title'].str.lower() == title.lower()]

    if not matching_row.empty:
        return matching_row['descr'].values[0]




