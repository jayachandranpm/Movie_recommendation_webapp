import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import fuzz, process  # Install fuzzywuzzy via pip

# Path to your CSV file containing movie metadata
csv_file_path = 'mymoviedb.csv'

# Load movie metadata from the CSV file
metadata = pd.read_csv(csv_file_path, low_memory=False)

# Clean the data and handle missing values
metadata['title'] = metadata['title'].fillna('')
metadata['overview'] = metadata['overview'].fillna('')

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata['overview'])

# Perform Truncated SVD to reduce dimensionality
n_components = 100  # Adjust the number of components as needed
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

# Calculate cosine similarity between movies using the reduced-dimensional TF-IDF matrix
cosine_sim = linear_kernel(tfidf_matrix_svd, tfidf_matrix_svd)

# Function to get top-N movie recommendations for a movie title
def get_top_n_recommendations(movie_title, n=10):
    # Convert the user input to title case
    movie_title = movie_title.title()

    # Check if the user input matches a movie title
    if movie_title not in metadata['title'].values:
        # Suggest movie titles based on user input
        suggestions = suggest_movie_titles(movie_title)
        return suggestions

    movie_index = metadata[metadata['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_n_recommendations = similarity_scores[1:n+1]  # Exclude the movie itself
    top_n_movie_indices = [index for index, _ in top_n_recommendations]
    top_n_movie_titles = metadata['title'].iloc[top_n_movie_indices].tolist()
    return top_n_movie_titles

# Function to suggest movie titles based on user input
def suggest_movie_titles(user_input, threshold=70, limit=5):
    # Use fuzzywuzzy to get movie title suggestions
    suggestions = process.extract(user_input, metadata['title'], limit=limit, scorer=fuzz.token_sort_ratio)
    return [suggestion[0] for suggestion in suggestions if suggestion[1] >= threshold]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title('🎬 Movie Recommendation System')
st.sidebar.header('Choose a Movie and Get Recommendations')

movie_title = st.sidebar.text_input('Enter a movie title:')
if st.sidebar.button('Get Recommendations'):
    recommendations = get_top_n_recommendations(movie_title)
    if not recommendations:
        st.warning(f"No recommendations found for '{movie_title}'. Try a different movie.")
    else:
        st.subheader(f'Top Movie Recommendations for {movie_title}')
        for i, recommended_movie in enumerate(recommendations):
            st.write(f'{i + 1}. {recommended_movie}')

    # Suggest movie titles based on user input
    movie_suggestions = suggest_movie_titles(movie_title, threshold=70, limit=5)
    if movie_suggestions:
        st.subheader('Suggested Movie Titles:')
        st.write(movie_suggestions)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .st-c3 {
        background-color: #f0f0f0;
    }
    .st-ag {
        background-color: #ffffff;
    }
    .st-c2 {
        color: black;
    }
    .st-df {
        font-size: 20px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
