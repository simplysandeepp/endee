"""Test script to debug vector insertion"""
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a simple test vector
text = "This is a test"
embedding = model.encode([text])[0]
vector_data = embedding.astype(np.float32).tolist()

print(f"Vector dimension: {len(vector_data)}")
print(f"Vector type: {type(vector_data[0])}")

# Test 1: Single object
payload1 = {
    "id": "test1",
    "vector": vector_data,
    "meta": text
}

print("\n=== Test 1: Single object ===")
print(f"Payload: {payload1}")

response = requests.post(
    "http://localhost:8080/api/v1/index/RAG/vector/insert",
    json=payload1,
    timeout=5
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test 2: Array of objects
payload2 = [{
    "id": "test2",
    "vector": vector_data,
    "meta": text
}]

print("\n=== Test 2: Array of objects ===")

response = requests.post(
    "http://localhost:8080/api/v1/index/RAG/vector/insert",
    json=payload2,
    timeout=5
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
