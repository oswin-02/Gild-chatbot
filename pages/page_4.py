# pages/cbow_vs_skipgram.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import nltk
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

# Set page config
st.set_page_config(page_title="Skip-gram vs CBOW Word2Vec", layout="wide")

# Title
st.title("Comparison of Skip-gram and CBOW Word2Vec Models")

# Parameters
vector_size = 100
window_size = 5
min_count = 5
workers = 4

# Download Brown corpus
nltk.download('brown')
sentences = nltk.corpus.brown.sents()

# Preprocess sentences
tokenized_sentences = [
    [word for word in simple_preprocess(remove_stopwords(" ".join(sentence)))]
    for sentence in sentences
]

# Train models
skip_gram_model = Word2Vec(
    tokenized_sentences,
    vector_size=vector_size,
    window=window_size,
    min_count=min_count,
    workers=workers,
    sg=1  # Skip-gram
)
cbow_model = Word2Vec(
    tokenized_sentences,
    vector_size=vector_size,
    window=window_size,
    min_count=min_count,
    workers=workers,
    sg=0  # CBOW
)

# Extract word vectors
sg_japan_vector = skip_gram_model.wv['japan']
cbow_japan_vector = cbow_model.wv['japan']

# Similar words
sg_similar_words = skip_gram_model.wv.most_similar('japan')
cbow_similar_words = cbow_model.wv.most_similar('japan')

# Similarity between japan and russia
sg_similarity = skip_gram_model.wv.similarity('japan', 'russia')
cbow_similarity = cbow_model.wv.similarity('japan', 'russia')

# Layout
st.header("Vectors for 'japan' (first 10 elements)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Skip-gram Model")
    st.write(sg_japan_vector[:10])
    st.subheader("Most Similar Words")
    st.table(sg_similar_words)

with col2:
    st.subheader("CBOW Model")
    st.write(cbow_japan_vector[:10])
    st.subheader("Most Similar Words")
    st.table(cbow_similar_words)

# Similarity Scores
st.header("Similarity Between 'japan' and 'russia'")
similarity_df = pd.DataFrame({
    'Model': ['Skip-gram', 'CBOW'],
    'Similarity': [sg_similarity, cbow_similarity]
})
st.bar_chart(similarity_df.set_index('Model'))

# Scatter plot
st.header("Scatter Plot of Word Embeddings (First 2 Dimensions)")

df = pd.DataFrame({
    'Model': ['Skip-gram', 'CBOW', 'Skip-gram', 'CBOW'],
    'Word': ['japan', 'japan', 'russia', 'russia'],
    'X': [skip_gram_model.wv['japan'][0], cbow_model.wv['japan'][0],
          skip_gram_model.wv['russia'][0], cbow_model.wv['russia'][0]],
    'Y': [skip_gram_model.wv['japan'][1], cbow_model.wv['japan'][1],
          skip_gram_model.wv['russia'][1], cbow_model.wv['russia'][1]]
})

fig = px.scatter(
    df,
    x='X',
    y='Y',
    color='Model',
    hover_name='Word',
    title="Word Embedding Scatter Plot: Skip-gram vs CBOW",
)
st.plotly_chart(fig)
