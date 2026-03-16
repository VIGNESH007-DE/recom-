import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Load CSS ----------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Recommendation Bot",
    page_icon="🎬",
    layout="centered"
)

# ---------- Title ----------
st.title("🎬 Netflix  Movie Recommendation")
st.write("Select a movie and get similar movie recommendations")

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies = movies.dropna()
    return movies

movies = load_data()

# ---------- Recommendation Model ----------
cv = CountVectorizer(tokenizer=lambda x: x.split('|'))

genre_matrix = cv.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(genre_matrix)

movies = movies.reset_index()

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ---------- Recommendation Function ----------
def recommend(movie_name):

    idx = indices[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]


# ---------- Movie Selection ----------
movie_list = movies['title'].values

selected_movie = st.selectbox("🎥 Select a Movie", movie_list)

# ---------- Recommendation Button ----------
if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("🍿 Recommended Movies")

    for movie in recommendations:
        st.write("✔", movie)

# ---------- Footer ----------
st.markdown("---")
st.write("Built using Streamlit Movie Recommendation System")
