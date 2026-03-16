import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.title("🎬 Movie Recommendation System")

movies = pd.read_csv("movies.csv")

cv = CountVectorizer(tokenizer=lambda x: x.split('|'))

genre_matrix = cv.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(genre_matrix)

movies = movies.reset_index()

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_name):

    idx = indices[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]

movie_list = movies['title'].values

selected_movie = st.selectbox("Select Movie", movie_list)

if st.button("Recommend"):

    rec = recommend(selected_movie)

    st.write("### Recommended Movies")

    for i in rec:
        st.write(i)
