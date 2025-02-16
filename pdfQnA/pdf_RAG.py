# Install required packages
# pip install langchain openai transformers sentence-transformers faiss-cpu streamlit redis

import faiss
import openai
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import redis
from openai.error import OpenAIError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API key (replace with your key)
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize Redis cache
cache = redis.Redis()

# Load and split documents
def load_and_split_documents(file_path):
    # Load the document (PDF in this case)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Chunk documents for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Print chunk structures for debugging
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")  # Debug print to check structure
    
    return chunks

# Embed the chunks using SentenceTransformer
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a small transformer for speed

    # Ensure chunks have the correct structure
    try:
        # Attempt to extract text from each chunk
        embeddings = model.encode([chunk.page_content for chunk in chunks])  # Change 'text' to 'page_content'
    except Exception as e:
        print(f"Error during embedding: {e}")
        return None
    return embeddings

# Create FAISS index for similarity search
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Get the embedding dimension
    index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(embeddings)
    return index

# Retrieve top K similar documents using FAISS
def retrieve_similar_documents(query, index, chunks, embedder, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)  # Get top k results
    results = [chunks[i].page_content for i in I[0]]  # Change 'text' to 'page_content'
    return results

# Generate a response using OpenAI GPT-3.5 based on the retrieved documents
def generate_response(query, retrieved_docs):
    prompt = f"Answer the question based on the following documents:\n\n{retrieved_docs}\n\nQuestion: {query}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        return "Sorry, there was an error processing your request."

# Cache query responses using Redis
def get_cached_response(query):
    return cache.get(query)

def set_cached_response(query, result):
    cache.set(query, result)

# Streamlit frontend
st.title("RAG-based Question Answering System")

# Upload a document (PDF format)
uploaded_file = st.file_uploader("Upload a document (PDF)", type="pdf")

if uploaded_file:
    # Load and split the document
    with open("uploaded_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Processing document...")
    chunks = load_and_split_documents("uploaded_doc.pdf")
    
    if chunks:  # Proceed only if chunks were created
        embeddings = embed_chunks(chunks)
        
        if embeddings is not None:  # Proceed only if embeddings were created
            index = create_faiss_index(embeddings)
            embedder = SentenceTransformer('all-MiniLM-L6-v2')

            # Query input
            query = st.text_input("Enter your query")

            if query:
                # Check cache for previous response
                cached_response = get_cached_response(query)
                if cached_response:
                    st.write(f"Cached Response: {cached_response.decode()}")
                else:
                    st.write("Retrieving relevant information...")
                    # Retrieve top K similar documents
                    retrieved_docs = retrieve_similar_documents(query, index, chunks, embedder)

                    # Generate answer from OpenAI
                    st.write("Generating response...")
                    answer = generate_response(query, "\n".join(retrieved_docs))

                    # Cache the answer
                    set_cached_response(query, answer)

                    # Display the generated answer
                    st.write(f"Generated Response: {answer}")
        else:
            st.write("Error: Could not create embeddings from the document chunks.")
