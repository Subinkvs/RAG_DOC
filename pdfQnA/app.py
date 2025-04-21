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


@app.route('/document', methods=['POST'])
def upload_and_query():
    global stored_index, stored_chunks

    # 1. Read query (from form-data) or JSON
    query = request.form.get("query") or (request.get_json() or {}).get("query")
    if not query:
        return jsonify({"error": "Missing `query` parameter"}), 400

    # 2. If a file is provided, rebuild index
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        temp_path = "tmp_upload.pdf"
        try:
            file.save(temp_path)
            chunks = load_and_split_documents(temp_path)
            embs = embed_chunks(chunks)
            stored_index = create_faiss_index(embs)
            stored_chunks = chunks
        except Exception as e:
            return jsonify({"error": f"Document processing failed: {e}"}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 3. Ensure we've got an index
    if stored_index is None or stored_chunks is None:
        return jsonify({"error": "No document indexed. Please include a PDF file in your request."}), 400

    # 4. Try cache
    cached = cache.get(query)
    if cached:
        return jsonify({"response": cached.decode()}), 200

    # 5. Retrieval + generation
    try:
        docs = retrieve_similar_documents(query, stored_index, stored_chunks)
        answer = generate_response(query, "\n".join(docs))
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {e}"}), 500

    # 6. Cache and return
    cache.set(query, answer)
    return jsonify({"response": answer}), 200


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
