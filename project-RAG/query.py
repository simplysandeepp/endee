"""
Query system - converts questions to embeddings, searches Endee, generates answers
"""
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

class QueryEngine:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.endee_url = ENDEE_URL
        self.index_name = INDEX_NAME
        
    def create_query_embedding(self, question):
        """Convert question to embedding"""
        embedding = self.model.encode(question)
        return embedding
    
    def search_endee(self, query_embedding, top_k=TOP_K):
        """Search similar vectors in Endee"""
        url = f"{self.endee_url}/api/v1/index/{self.index_name}/search"
        
        payload = {
            "vector": query_embedding.tolist(),
            "k": top_k
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            results = response.json()
            return results
        else:
            print(f"✗ Search failed: {response.text}")
            return None
    
    def retrieve_documents(self, search_results):
        """Extract relevant text from search results"""
        documents = []
        
        if search_results and "results" in search_results:
            for result in search_results["results"]:
                if "metadata" in result and "text" in result["metadata"]:
                    documents.append({
                        "text": result["metadata"]["text"],
                        "score": result.get("distance", 0)
                    })
        
        return documents
    
    def generate_answer(self, question, documents):
        """Generate answer using retrieved context (simple concatenation)"""
        if not documents:
            return "No relevant documents found."
        
        # Combine retrieved documents as context
        context = "\n\n".join([doc["text"] for doc in documents])
        
        # Simple answer generation (in production, use LLM like GPT/Claude)
        answer = f"Based on the documents:\n\n{context}\n\n"
        answer += f"Answer to '{question}':\n"
        answer += "The relevant information has been retrieved from the knowledge base."
        
        return answer
    
    def query(self, question):
        """Complete query pipeline"""
        print(f"\n❓ Question: {question}\n")
        
        # Create embedding
        query_embedding = self.create_query_embedding(question)
        print(f"✓ Created query embedding")
        
        # Search Endee
        search_results = self.search_endee(query_embedding)
        print(f"✓ Searched vector database")
        
        # Retrieve documents
        documents = self.retrieve_documents(search_results)
        print(f"✓ Retrieved {len(documents)} relevant documents\n")
        
        # Display retrieved documents
        print("📚 Retrieved Documents:")
        for i, doc in enumerate(documents, 1):
            print(f"\n[Document {i}] (Score: {doc['score']:.4f})")
            print(doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'])
        
        # Generate answer
        answer = self.generate_answer(question, documents)
        print(f"\n💡 Answer:\n{answer}\n")
        
        return answer

if __name__ == "__main__":
    engine = QueryEngine()
    
    # Example queries
    questions = [
        "What is machine learning?",
        "Explain deep learning",
        "What are neural networks?"
    ]
    
    for question in questions:
        engine.query(question)
