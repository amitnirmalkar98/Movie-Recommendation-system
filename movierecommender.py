import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load your dataset
movies_df = pd.read_csv('movies.csv')

# Fill missing values
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')

# Combine features for the recommendation system
movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['overview'] + ' ' + movies_df['cast'] + ' ' + movies_df['director']

# TF-IDF Vectorizer and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, movie_indices=movie_indices):
    if title in movie_indices:
        idx = movie_indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices_list = [i[0] for i in sim_scores]
        return movies_df['title'].iloc[movie_indices_list].tolist()
    else:
        return ["Title not found"]

# Initialize the GUI window
window = tk.Tk()
window.title("Movie Recommendation App")
window.geometry("800x500")

# Dropdown and button
movie_titles = movies_df['title'].tolist()
selected_movie = tk.StringVar()
dropdown = ttk.Combobox(window, textvariable=selected_movie, values=movie_titles, width=50)
dropdown.set("Select a movie")
dropdown.pack(pady=10)

def show_recommendations():
    movie = selected_movie.get()
    recommendations = get_recommendations(movie)
    result_label.config(text="Recommended Movies:\n" + "\n".join(recommendations))

recommend_button = tk.Button(window, text="Get Recommendations", command=show_recommendations)
recommend_button.pack(pady=20)
result_label = tk.Label(window, text="", wraplength=800, justify="left")
result_label.pack(pady=10)

window.mainloop()
