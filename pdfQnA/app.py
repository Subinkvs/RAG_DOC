from flask import Flask, request, jsonify
import faiss
import openai
import redis
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.error import OpenAIError
import os
from dotenv import load_dotenv
from flask_cors import CORS


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
cache = redis.Redis()

# Retrieve API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")
openai.api_key = openai_api_key

# Initialize global variables
stored_index = None
stored_chunks = None
stored_embedder = None

# Load and split documents
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Embed the chunks
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([chunk.page_content for chunk in chunks])

# Create FAISS index
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Retrieve top K similar documents
def retrieve_similar_documents(query, index, chunks, embedder, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i].page_content for i in I[0]]

# Generate response using OpenAI
def generate_response(query, retrieved_docs):
    prompt = f"Answer the question based on the following documents:\n\n{retrieved_docs}\n\nQuestion: {query}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except OpenAIError as e:
        return f"Error: {str(e)}"

# API to upload and process a PDF document efficently
@app.route('/upload', methods=['POST'])
def upload_document():
    global stored_index, stored_chunks, stored_embedder

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "uploaded_doc.pdf"
    file.save(file_path)

    chunks = load_and_split_documents(file_path)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    stored_index = index
    stored_chunks = chunks
    stored_embedder = SentenceTransformer('all-MiniLM-L6-v2')

    return jsonify({"message": "Document uploaded and processed successfully"}), 200

# API to query the document
@app.route('/query', methods=['POST'])
def query_document():
    global stored_index, stored_chunks, stored_embedder

    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if stored_index is None:
        return jsonify({"error": "No document uploaded. Please upload a document first."}), 400

    cached_response = cache.get(query)
    if cached_response:
        return jsonify({"response": cached_response.decode()})

    retrieved_docs = retrieve_similar_documents(query, stored_index, stored_chunks, stored_embedder)
    answer = generate_response(query, "\n".join(retrieved_docs))

    cache.set(query, answer)
    return jsonify({"response": answer})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

