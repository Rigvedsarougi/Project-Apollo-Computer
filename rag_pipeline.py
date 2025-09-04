# rag_pipeline.py
import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai
from typing import List
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="RAG Pipeline with DeepSeek",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class DeepSeekAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 1000) -> str:
        """Generate response using DeepSeek API"""
        
        # Combine context and prompt
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Use the provided context to answer the user's question. If the context doesn't contain the answer, say so politely."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def load_and_process_documents(self, uploaded_files) -> List[str]:
        """Load and process uploaded PDF documents"""
        documents = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_file_path)
        
        # Split documents into chunks
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        return []
    
    def create_vector_store(self, chunks):
        """Create vector store from document chunks"""
        if chunks:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            vector_store.persist()
            return vector_store
        return None

def main():
    st.title("🤖 RAG Pipeline with DeepSeek API")
    st.markdown("Upload PDF documents, then ask questions about their content.")
    
    # Initialize components
    processor = DocumentProcessor()
    
    # Sidebar for API key and document upload
    with st.sidebar:
        st.header("Configuration")
        
        # API key input
        api_key = st.text_input("DeepSeek API Key", type="password")
        if api_key:
            deepseek_api = DeepSeekAPI(api_key)
            st.success("API key configured")
        else:
            st.info("Please enter your DeepSeek API key to continue")
            deepseek_api = None
        
        # Document upload
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if uploaded_files and deepseek_api:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    chunks = processor.load_and_process_documents(uploaded_files)
                    if chunks:
                        st.session_state.vector_store = processor.create_vector_store(chunks)
                        st.session_state.documents_processed = True
                        st.success(f"Processed {len(chunks)} document chunks")
                    else:
                        st.error("Failed to process documents")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Ask a Question")
        
        if not st.session_state.documents_processed:
            st.info("Please upload and process documents first")
        else:
            query = st.text_input("Enter your question:")
            
            if query and deepseek_api:
                if st.button("Get Answer"):
                    with st.spinner("Searching for answer..."):
                        # Retrieve relevant documents
                        relevant_docs = st.session_state.vector_store.similarity_search(query, k=3)
                        context = "\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Generate response using DeepSeek API
                        response = deepseek_api.generate_response(query, context)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": response,
                            "context": context
                        })
    
    with col2:
        st.header("Chat History")
        
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"Q: {chat['question']}"):
                    st.markdown(f"**A:** {chat['answer']}")
                    with st.expander("View context used"):
                        st.text(chat['context'])
        else:
            st.info("Your questions and answers will appear here")
    
    # Display sample queries if no chat history
    if not st.session_state.chat_history and st.session_state.documents_processed:
        st.divider()
        st.subheader("Sample Queries")
        
        sample_queries = [
            "What are the key concepts discussed in these documents?",
            "Can you summarize the main points?",
            "What are the conclusions or recommendations?",
            "Are there any important definitions I should know?",
            "What methods or approaches are described?"
        ]
        
        for query in sample_queries:
            if st.button(f"Ask: {query}", key=f"sample_{query}"):
                with st.spinner("Searching for answer..."):
                    # Retrieve relevant documents
                    relevant_docs = st.session_state.vector_store.similarity_search(query, k=3)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Generate response using DeepSeek API
                    response = deepseek_api.generate_response(query, context)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": response,
                        "context": context
                    })
                    st.rerun()

if __name__ == "__main__":
    main()
