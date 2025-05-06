# Full updated code with document-specific querying using document_id

from flask import Flask, request, jsonify
import faiss
import openai
import redis
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.error import OpenAIError
import os
import json
import uuid
import pickle
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Connect to Redis
cache = redis.Redis()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# Load and split PDF document
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Embed chunks using SentenceTransformer
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([chunk.page_content for chunk in chunks])

# Create FAISS index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Search top-K similar documents
def retrieve_similar_documents(query, index, chunks, embedder, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i].page_content for i in I[0]]

# Ask OpenAI to answer using retrieved docs
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

# Ask OpenAI to extract metadata
def extract_metadata_with_openai(text):
    prompt = (
        "Extract metadata like Project Code, Project Name, Originator, Document Type, Discipline, Sub Discipline, "
        "Document Title, Version/Revision, Location, Floor Number from the following text:\n\n"
        f"{text}\n\nReturn JSON."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        metadata_json = response['choices'][0]['message']['content'].strip()
        return json.loads(metadata_json)
    except (OpenAIError, json.JSONDecodeError) as e:
        return {"error": f"Metadata extraction error: {str(e)}"}

# --- API Endpoints ---

# Upload document and extract metadata
@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = "uploaded_doc.pdf"
        file.save(file_path)

        chunks = load_and_split_documents(file_path)
        embeddings = embed_chunks(chunks)
        index = create_faiss_index(embeddings)

        doc_text = "\n".join([chunk.page_content for chunk in chunks])
        metadata = extract_metadata_with_openai(doc_text)

        document_id = str(uuid.uuid4())
        cache.setex(f"metadata:{document_id}", 86400, json.dumps(metadata))
        cache.set(f"index:{document_id}", pickle.dumps(index))
        cache.set(f"chunks:{document_id}", pickle.dumps(chunks))

        return jsonify({"message": "Document uploaded successfully", "document_id": document_id}), 200

    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Get extracted metadata by document_id
@app.route('/metadata', methods=['GET'])
def get_metadata():
    document_id = request.args.get('document_id')
    if not document_id:
        return jsonify({"error": "Missing document_id"}), 400

    metadata_json = cache.get(f"metadata:{document_id}")
    if metadata_json is None:
        return jsonify({"error": "No metadata found for this document_id"}), 404

    metadata = json.loads(metadata_json)
    return jsonify({"metadata": metadata}), 200

# Edit extracted metadata
@app.route('/metadata/edit', methods=['POST'])
def edit_metadata():
    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON input"}), 400

        data = request.get_json()
        document_id = data.get('document_id')
        updates = data.get('updates')

        if not document_id or not updates:
            return jsonify({"error": "Provide both document_id and updates"}), 400

        metadata_json = cache.get(f"metadata:{document_id}")
        if metadata_json is None:
            return jsonify({"error": "No metadata found for this document_id"}), 404

        metadata = json.loads(metadata_json)

        for key, value in updates.items():
            if key in metadata:
                metadata[key] = value

        cache.setex(f"metadata:{document_id}", 86400, json.dumps(metadata))

        return jsonify({"message": "Metadata updated", "updated_metadata": metadata}), 200

    except Exception as e:
        return jsonify({"error": f"Edit failed: {str(e)}"}), 500

# Query document for answers (now supports document_id)
@app.route('/query', methods=['POST'])
def query_document():
    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON"}), 400

        data = request.get_json()
        query = data.get('query')
        document_id = data.get('document_id')

        if not query or not document_id:
            return jsonify({"error": "Both 'query' and 'document_id' are required"}), 400

        index_data = cache.get(f"index:{document_id}")
        chunks_data = cache.get(f"chunks:{document_id}")
        if index_data is None or chunks_data is None:
            return jsonify({"error": "Document data not found for this document_id"}), 404

        index = pickle.loads(index_data)
        chunks = pickle.loads(chunks_data)
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        retrieved_docs = retrieve_similar_documents(query, index, chunks, embedder, k=5)
        response = generate_response(query, "\n".join(retrieved_docs))

        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

# --- Run Server ---
if __name__ == "__main__":
    app.run(debug=True)

