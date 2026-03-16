import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
movies = pd.read_csv("movies.csv")

# Fill missing values
movies = movies.dropna()

# Use genres for similarity
cv = CountVectorizer(tokenizer=lambda x: x.split('|'))

genre_matrix = cv.fit_transform(movies['genres'])

# Calculate similarity
cosine_sim = cosine_similarity(genre_matrix)

# Reset index
movies = movies.reset_index()

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_name):
    
    idx = indices[movie_name]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices]
