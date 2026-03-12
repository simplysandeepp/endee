# Simple RAG Application with Endee Vector Database

A minimal RAG (Retrieval-Augmented Generation) system using:
- **Endee** for vector storage and semantic search
- **Groq** for fast LLM responses
- **Streamlit** for simple web interface

## 🚀 Live Demo

[Coming soon - Deploy to Streamlit Cloud]

## ✨ Features

- Upload text documents and automatically chunk them
- Semantic search using sentence transformers
- Fast AI-powered answers using Groq LLM
- Simple and clean web interface
- Powered by Endee vector database

## 📋 Prerequisites

- Python 3.8+
- Endee vector database running (locally or remote)
- Groq API key

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project-RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_actual_groq_api_key
ENDEE_URL=http://localhost:8080
INDEX_NAME=RAG
```

### 4. Start Endee vector database

Make sure Endee is running on `http://localhost:8080`

For local Endee setup, see: [Endee Documentation](https://github.com/endee-io/endee)

### 5. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📦 Deployment to Streamlit Cloud

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - RAG application"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `project-RAG/app.py`
6. Click "Advanced settings" and add secrets:
   ```
   GROQ_API_KEY = "your_groq_api_key"
   ENDEE_URL = "your_endee_url"
   INDEX_NAME = "RAG"
   ```
7. Click "Deploy"

**Note:** For deployment, you'll need a publicly accessible Endee instance. You can:
- Deploy Endee on a cloud server (AWS, GCP, Azure)
- Use Docker to deploy Endee
- Use a tunnel service like ngrok for testing

## 🎯 Usage

1. **Initialize Index**: Click "Initialize Index" in the sidebar (first time only)
2. **Upload Document**: Upload a `.txt` file
3. **Ingest**: Click "Ingest Document" to process and store vectors
4. **Ask Questions**: Type your question and get AI-powered answers based on your documents

## 📝 Example

Upload a document about a village story, then ask:
- "Who was the engineer living near the banyan tree?"
- "What problem did the village face?"
- "How did they solve the drought issue?"

## 🏗️ Architecture

```
User Question → Embedding → Endee Search → Context → Groq LLM → Answer
```

## 🔧 Configuration

Edit `.env` to customize:
- `GROQ_API_KEY`: Your Groq API key
- `ENDEE_URL`: Endee server URL
- `INDEX_NAME`: Vector index name

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📞 Support

For issues or questions:
- Endee: [https://github.com/endee-io/endee](https://github.com/endee-io/endee)
- Groq: [https://console.groq.com](https://console.groq.com)

