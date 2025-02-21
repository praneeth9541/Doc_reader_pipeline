import streamlit as st
import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
try:
    client = QdrantClient("localhost", port=6333)
    st.success("Connected to Qdrant server!")
except Exception as e:
    st.error(f"Failed to connect to Qdrant server: {e}")
    st.stop()

# Streamlit App Title
st.title("Document Processing Pipeline")
st.write("Upload a PDF, enter a query, and see the pipeline output!")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Step 2: Extract Text from PDF
    st.header("Step 1: Extracted Text")
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    st.text_area("Extracted Text", text, height=200)

    # Step 3: Chunk Text Dynamically
    st.header("Step 2: Chunked Text")
    def chunk_text(text, max_length=512):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    chunks = chunk_text(text)
    st.write(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        st.write(f"**Chunk {i+1}:** {chunk}")

    # Step 4: Generate Embeddings
    st.header("Step 3: Generated Embeddings")
    embeddings = model.encode(chunks)
    st.write(f"Embeddings shape: {embeddings.shape}")

    # Step 5: Perform Vector Search
    st.header("Step 4: Vector Search")
    query = st.text_input("Enter your query:")
    if query:
        query_embedding = model.encode(query)
        st.write(f"Query embedding shape: {query_embedding.shape}")

        # Check for NaN values
        if np.isnan(query_embedding).any():
            st.error("Query embedding contains NaN values!")
            st.stop()

        # Create Qdrant collection (if not already created)
        try:
            client.recreate_collection(
                collection_name="document_chunks",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except Exception as e:
            st.warning(f"Collection already exists or could not be created: {e}")

        # Add embeddings to Qdrant
        client.upsert(
            collection_name="document_chunks",
            points=[
                {"id": idx, "vector": embedding.tolist(), "payload": {"text": chunk}}
                for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
            ]
        )

        # Perform vector search
        search_result = client.search(
            collection_name="document_chunks",
            query_vector=query_embedding.tolist(),
            limit=5
        )

        st.write("**Top 5 Search Results:**")
        for result in search_result:
            st.write(f"**Score:** {result.score}, **Text:** {result.payload['text']}")