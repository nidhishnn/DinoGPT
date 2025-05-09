# streamlit_app.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import numpy as np
import faiss

# Must be first
st.set_page_config(page_title="DinoGPT", layout="wide")

# === Load Models ===
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("text2text-generation", model="google/flan-t5-small")
    return embed_model, qa_model

# === Load Data + FAISS Index ===
@st.cache_data
def load_data():
    df = pd.read_csv("dinosaur_with_descriptions.csv")
    embeddings = np.load("dino_embeddings.npy").astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return df, embeddings, index

embed_model, qa_model = load_models()
df, dino_embeddings, index = load_data()

# === Hero Section ===
st.markdown("<h1 style='text-align: center; font-size: 48px;'>DinoGPT</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 18px;'>
Your personal dinosaur database RAG query engine.
<br>
</div>
""", unsafe_allow_html=True)

# === Image Display (Centered, Fixed Size) ===
from PIL import Image

def resize_fixed(img, size=(400, 300)):
    return img.resize(size)

tricera_img = resize_fixed(Image.open("tricera.jpg"))
stego_img = resize_fixed(Image.open("stegosaurus.jpg"))

# 5 columns: empty | Tricera | spacer | Stego | empty
col_empty1, col_tricera, col_spacer, col_stego, col_empty2 = st.columns([1, 3, 1, 3, 1])

with col_tricera:
    st.image(tricera_img, caption="Triceratops")

with col_stego:
    st.image(stego_img, caption="Stegosaurus")



# === System Overview ===
st.markdown("""
---  
### How does DinoGPT work?
DinoGPT is a Retrieval-Augmented Generation (RAG) system that combines semantic vector search with large language models (LLMs) to provide accurate and contextually relevant answers to your natural language dinosaur-related queries. Here are the key components:
1. **Dataset**: Kaggle Dinosaur Dataset (https://www.kaggle.com/datasets/smruthiiii/dinosaur-dataset). The dataset contains features for1042 unique dinosaurs.        
2. **Textual Descriptions**: Additional textual descriptions for each dinosaur are scraped from Wikipedia using its public API.
3. **Embeddings**: Each dinosaur description is transformed into a 384-dimensional semantic vector using the `all-MiniLM-L6-v2` model from SentenceTransformers. 
4. **Vector Storage**: Embeddings are stored in a FAISS index using the 'IndexFlatIP' method for efficient similarity search.
5. **Query Processing**: When you ask a question, it is converted into a vector using the same embedding model. 
6. **Similarity Search**: The FAISS index is used to find the top-k most similar dinosaur descriptions based on the cosine similarity between the query vector and the stored vectors.      
7. **LLM Answer Generation**: The FLAN-T5 (Google) model is prompted with the user's query and the retrieved dinosaur descriptions to provide a tailored, context-specific answer. 
---
""")

# === Query Section ===
st.subheader("DinoGPT Query Input")
user_query = st.text_input("What would you like to know?", "")
top_k = st.slider("Number of similar dinosaurs to retrieve", min_value=1, max_value=10, value=3)

# === RAG Pipeline ===
if st.button("Get Answer") and user_query:
    with st.spinner("Searching fossil records..."):
        query_vec = embed_model.encode([user_query], normalize_embeddings=True).astype('float32')
        _, inds = index.search(query_vec, top_k)
        context = "\n\n".join(df.iloc[i]["full_description"] for i in inds[0])

        prompt = f"""Use the following dinosaur descriptions to answer the question.

{context}

Question: {user_query}
Answer:"""

        response = qa_model(prompt, max_new_tokens=300)[0]['generated_text'].strip()

        st.markdown("### Answer")
        st.markdown(response)

        with st.expander("See Retrieved Dinosaur Descriptions"):
            st.text(context)
