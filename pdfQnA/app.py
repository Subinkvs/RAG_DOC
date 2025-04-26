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
import re
from docx import Document as DocxDocument
import pandas as pd
import json


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Explicitly allow NestJS app to access this API
CORS(app, resources={r"/*": {"origins": "http://localhost:3000",  # Adjust if NestJS runs on a different port
                             "methods": ["GET", "POST", "OPTIONS"],
                             "allow_headers": ["Content-Type", "Authorization"]}})

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
document_metadata = {}

from langchain.docstore.document import Document

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

# Use OpenAI to extract metadata from document text
def extract_metadata_with_openai(text):
    prompt = (
        "Extract key metadata such as Project Code, Project Name, Originator, Document Type, Discipline, "
        "Sub Discipline, Document Title, Version/Revision, Location, and Floor Number from the following text:\n\n"
        f"{text}\n\nReturn the metadata in a JSON format."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        metadata_str = response['choices'][0]['message']['content'].strip()
        return json.loads(metadata_str)  # Convert JSON string to Python dict
    except OpenAIError as e:
        return {"error": f"Error extracting metadata: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse metadata JSON: {str(e)}"}


@app.route('/upload', methods=['POST'])
def upload_document():
    global stored_index, stored_chunks, stored_embedder, document_metadata

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = "uploaded_doc.pdf"
        file.save(file_path)

        # Process the uploaded file
        try:
            chunks = load_and_split_documents(file_path)
            embeddings = embed_chunks(chunks)
            index = create_faiss_index(embeddings)

            # Extract text from document chunks for metadata extraction
            doc_text = "\n".join([chunk.page_content for chunk in chunks])

            # Use OpenAI to extract metadata
            document_metadata = extract_metadata_with_openai(doc_text)

        except Exception as processing_error:
            return jsonify({"error": f"Document processing failed: {str(processing_error)}"}), 500

        # Store the processed components
        stored_index = index
        stored_chunks = chunks
        stored_embedder = SentenceTransformer('all-MiniLM-L6-v2')

        return jsonify({"message": "Document uploaded and processed successfully"}), 200

    except IOError as io_err:
        return jsonify({"error": f"I/O error occurred: {str(io_err)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    finally:
        # Optional cleanup: remove temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/metadata', methods=['GET'])
def get_metadata():
    # Return captured metadata
    if not document_metadata:
        return jsonify({"error": "No document metadata available. Please upload a document first."}), 400

    return jsonify({"metadata": document_metadata}), 200

@app.route('/metadata/edit', methods=['POST'])
def edit_metadata():
    global document_metadata

    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        # Allow users to edit metadata
        for key in data:
            if key in document_metadata:
                document_metadata[key] = data[key]

        return jsonify({"message": "Metadata updated successfully", "updated_metadata": document_metadata}), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query_document():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        if stored_index is None or stored_chunks is None:
            return jsonify({"error": "No documents have been uploaded. Please upload a document first."}), 400

        # Retrieve top k similar documents
        retrieved_docs = retrieve_similar_documents(query, stored_index, stored_chunks, stored_embedder, k=5)

        # Generate a response using OpenAI
        response = generate_response(query, "\n".join(retrieved_docs))

        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# # Run Flask app
# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5000)
