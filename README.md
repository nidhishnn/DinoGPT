# DinoGPT
![tricera](https://github.com/user-attachments/assets/b7c50cd8-4c61-41cf-be35-107951c715b9)

DinoGPT is a retrieval-augmented generation (RAG) tool that combines semantic vector search with large language models (LLMs) to deliver accurate, contextually relevant answers to your natural language dinosaur-related questions. 

**How it works**

1. Dataset: Kaggle Dinosaur Dataset (https://www.kaggle.com/datasets/smruthiiii/dinosaur-dataset). The dataset contains features for1042 unique dinosaurs.
2. Textual Descriptions: Additional textual descriptions for each dinosaur are scraped from Wikipedia using its public API.
3. Embeddings: Each dinosaur description is transformed into a 384-dimensional semantic vector using the all-MiniLM-L6-v2 model from SentenceTransformers.
4. Vector Storage: Embeddings are stored in a FAISS index using the 'IndexFlatIP' method for efficient similarity search.
5. Query Processing: When you ask a question, it is converted into a vector using the same embedding model.
6. Similarity Search: The FAISS index is used to find the top-k most similar dinosaur descriptions based on the cosine similarity between the query vector and the stored vectors.
7. LLM Answer Generation: The FLAN-T5 (Google) model is prompted with the user's query and the retrieved dinosaur descriptions to provide a tailored, context-specific answer.

**Demo**

Ask DinoGPT anything about dinosaurs -- fossil discovery locations, diets, genetic adaptations, sizes, or time periods -- and it will identify the most similar dinosaur descriptions to provide a tailored response.

Example <br>
Question: Where did Triceratops live? <br>
Answer: Western North America

DinoGPT further provides transparency into the retrieved dinosaur descriptions for the user to gain additional insight into the context surrounding the LLM's answer. As someone who has been fascinated by these remarkable creatures--especially Triceratops--that once roamed the Earth, this personal project reflects my ability to combine curiosity with cutting-edge AI tools. This app reflects my skills in leveraging RAG, building a user friendly interface in Streamlit, and exploring explainable natural language processing techniques.


