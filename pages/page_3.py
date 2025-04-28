

import streamlit as st
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

# Set page config
st.set_page_config(page_title="Skip-Gram Word2Vec Comparison", layout="wide")

# Title
st.title("Skip-Gram Word2Vec: With and Without Stopwords")

# Parameters
vector_size = 50
window_size = 10
min_count = 1
workers = 4
sg_flag = 1  # Skip-gram

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

# 1. WITHOUT removing stopwords
tokenized_sentences_normal = [simple_preprocess(sentence) for sentence in sentences]
model_normal = Word2Vec(
    tokenized_sentences_normal,
    vector_size=vector_size,
    window=window_size,
    min_count=min_count,
    workers=workers,
    sg=sg_flag
)
vector_normal = model_normal.wv['ukraine']
similar_normal = model_normal.wv.most_similar('ukraine')

# 2. WITH removing stopwords
tokenized_sentences_nostop = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]
model_nostop = Word2Vec(
    tokenized_sentences_nostop,
    vector_size=vector_size,
    window=window_size,
    min_count=min_count,
    workers=workers,
    sg=sg_flag
)
vector_nostop = model_nostop.wv['ukraine']
similar_nostop = model_nostop.wv.most_similar('ukraine')

# Display
st.header("Vector for 'ukraine' (first 10 elements)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Without Stopwords")
    st.write(vector_normal[:10])  # Only first 10 elements for readability
    st.subheader("Most Similar Words")
    st.table(similar_normal)

with col2:
    st.subheader("With Stopwords Removed")
    st.write(vector_nostop[:10])  # Only first 10 elements for readability
    st.subheader("Most Similar Words")
    st.table(similar_nostop)