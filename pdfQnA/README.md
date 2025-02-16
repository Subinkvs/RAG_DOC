# pdfQnA
## RAG Application 
The project will implement a RAG-based Question-Answering (QA) System that retrieves relevant documents from a knowledge base and generates a context-aware response using a language model. We'll focus on:

* Document Loading & Indexing (Chunking, Vector Embeddings, and Index Optimization)
* Retriever (BM25, Vector Search, or Hybrid Retrieval)
* Query Processing (Query Expansion and Rewriting)
* Language Model (Prompt Engineering, Fine-tuning, and Hallucination Mitigation)
* Caching, Async Processing, and Efficiency Optimization
* Answer Post-Processing (Answer Validation, Summarization)
* Monitoring & Human-in-the-Loop

## Tech Stack
* LangChain: For document loading, retrieval, and query management.
* Hugging Face or OpenAI GPT: For the generative language model.
* Elasticsearch or Pinecone: For document indexing and retrieval.
* Streamlit or FastAPI: To build a frontend or REST API interface.
* GPU: For model inference and optimization.

pip3 install langchain openai transformers sentence-transformers faiss-cpu pinecone-client elasticsearch streamlit

## Step-by-Step Breakdown of the Code:
Document Loading and Splitting:

### The script uses LangChain's PyPDFLoader to load a PDF document.
* The document is split into smaller chunks using the RecursiveCharacterTextSplitter class.

### Embeddings and Indexing:
* We use the SentenceTransformer model (all-MiniLM-L6-v2) to convert the document chunks into vector embeddings.
* The embeddings are indexed using FAISS, a fast similarity search library.

## Retrieval of Relevant Chunks:

* When the user submits a query, the query is embedded using the same SentenceTransformer.
* The FAISS index is used to retrieve the top k most relevant document chunks.

### OpenAI GPT-3.5 Response Generation:
The retrieved chunks are passed as context to the OpenAI GPT-3.5 model, which generates an answer based on the query and the retrieved chunks.

### Caching with Redis:

* The system caches the generated responses for repeated queries using Redis.
* If the same query is submitted, the cached response is returned instead of regenerating it.
Streamlit Interface:

* The user interface is built using Streamlit, allowing users to upload a PDF, submit queries, and view generated responses.

## Example Flow:
* User uploads a PDF document.
* The system loads and processes the document, creating chunks and embedding them for future queries.
* User submits a query (e.g., "What is the process of RAG?").
* The system checks if the query is cached. If cached, it returns the cached response.
* If not cached, the system retrieves the most relevant document chunks and uses GPT-3.5 to generate a response based on those chunks.
* The generated response is displayed to the user and cached for future queries.

## How to Run the Code:
* Set up OpenAI API Key: Replace "your-openai-api-key" with your actual OpenAI API key.
* Install Redis: Make sure you have Redis installed locally or hosted remotely.
* Run the Code: Use streamlit run your_script.py to launch the web interface locally.
