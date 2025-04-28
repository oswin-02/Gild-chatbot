import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Set page config
st.set_page_config(page_title="Word Embeddings Visualization", layout="wide")

# Title
st.title("2D Visualization of Word Embeddings")

# Sample sentences
sentences = [
    "Fierce fighting breaks out in Kyiv as Russian forces try to push their way towards the city centre from multiple directions.",
    "Responding to claims that the Russia is “ready for talks” with Ukraine, President Zelenskyy says his country is ready for peace talks with Russia, but not in Belarus.",
    "A planned evacuation from Mariupol and Volnovakha is thrown in chaos as Russia continues to attack despite agreeing a humanitarian corridor.",
    "In a video address to the Luxembourg parliament, President Zelenskyy reveals Russian forces currently occupy about 20% of Ukraine’s territory.",
    "The US Embassy in Moscow issues a security alert and urges American citizens to leave Russia immediately.",
    "President Putin signs “accession treaties” formalising Russia’s illegal annexation of four occupied regions in Ukraine, marking the largest forcible takeover of territory in Europe since the second world war.",
    "President Zelenskyy addresses the United Nations General Assembly in person for the first time since Russia began its invasion of his country in February 2022.",
    "The White House says North Korea has provided Russia with a shipment of weapons, calling it a troubling development and raising concerns about the expanded military relationship between the two countries.",
    "At a G7 meeting in Japan, the bloc’s foreign ministers insist that their support for Ukraine “will never waver”, despite growing tensions in the Middle East.",
    "The US suspends intelligence sharing with Kyiv."
]

# Preprocess the sentences
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word vectors
word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

# PCA for 2D reduction
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# Color map
color_map = {
    0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange',
    5: 'cyan', 6: 'magenta', 7: 'yellow', 8: 'pink', 9: 'brown'
}

# Assign colors
word_colors = []
for word in model.wv.index_to_key:
    for i, sentence in enumerate(tokenized_sentences):
        if word in sentence:
            word_colors.append(color_map[i])
            break

# Word IDs
word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

# Scatter plot for words
scatter = go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers+text',
    text=model.wv.index_to_key,
    textposition='top center',
    marker=dict(color=word_colors, size=8),
    customdata=word_colors,
    ids=word_ids,
    hovertemplate="Word: %{text}<br>Color: %{customdata}"
)

# Line traces for each sentence
line_traces = []
for i, sentence in enumerate(tokenized_sentences):
    line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv]
    if line_vectors:  # Only create a line if words exist
        line_trace = go.Scatter(
            x=[vec[0] for vec in line_vectors],
            y=[vec[1] for vec in line_vectors],
            mode='lines',
            line=dict(color=color_map[i], width=1),
            name=f"Sentence {i+1}"
        )
        line_traces.append(line_trace)

# Create figure
fig = go.Figure(data=[scatter] + line_traces)

fig.update_layout(
    title="2D Visualization of Word Embeddings",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    width=1000,
    height=1000,
    showlegend=True
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
