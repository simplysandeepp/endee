"""
Document ingestion pipeline - loads PDFs, creates embeddings, stores in Endee
"""
import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Configuration
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class DocumentIngestor:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.endee_url = ENDEE_URL
        self.index_name = INDEX_NAME
        
    def create_index(self):
        """Create Endee vector index"""
        url = f"{self.endee_url}/api/v1/index/create"
        payload = {
            "index_name": self.index_name,
            "dim": EMBEDDING_DIM,
            "space_type": "cosine",
            "M": 16,
            "ef_con": 200,
            "precision": "float32"
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"✓ Index '{self.index_name}' created successfully")
        elif "already exists" in response.text:
            print(f"✓ Index '{self.index_name}' already exists")
        else:
            print(f"✗ Failed to create index: {response.text}")
            
    def load_pdf(self, pdf_path):
        """Extract text from PDF"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_embeddings(self, chunks):
        """Convert text chunks to embeddings"""
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def store_in_endee(self, chunks, embeddings):
        """Store embeddings in Endee vector database"""
        url = f"{self.endee_url}/api/v1/index/{self.index_name}/insert"
        
        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": idx,
                "vector": embedding.tolist(),
                "metadata": {"text": chunk}
            })
        
        payload = {"vectors": vectors}
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print(f"✓ Stored {len(vectors)} vectors in Endee")
        else:
            print(f"✗ Failed to store vectors: {response.text}")
    
    def ingest_document(self, pdf_path):
        """Complete ingestion pipeline"""
        print(f"\n📄 Processing: {pdf_path}")
        
        # Load PDF
        text = self.load_pdf(pdf_path)
        print(f"✓ Extracted {len(text)} characters")
        
        # Chunk text
        chunks = self.chunk_text(text)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        print(f"✓ Generated embeddings (shape: {embeddings.shape})")
        
        # Store in Endee
        self.store_in_endee(chunks, embeddings)
        print("✓ Ingestion complete!\n")

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    
    # Create index
    ingestor.create_index()
    
    # Ingest all PDFs in data folder
    data_dir = "data"
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            ingestor.ingest_document(pdf_path)
