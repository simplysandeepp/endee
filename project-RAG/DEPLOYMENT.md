# Deployment Guide

## Option 1: Streamlit Cloud (Recommended for Demo)

### Prerequisites
- GitHub account
- Groq API key
- Publicly accessible Endee instance

### Steps

1. **Push to GitHub**
   ```bash
   cd project-RAG
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy Endee Publicly**
   
   You need Endee accessible from the internet. Options:
   
   **Option A: Deploy on Cloud (Recommended)**
   - Use Railway, Render, or Fly.io
   - Deploy using Docker
   - Get the public URL
   
   **Option B: Use ngrok (Testing only)**
   ```bash
   # In WSL where Endee is running
   ngrok http 8080
   # Copy the https URL (e.g., https://abc123.ngrok.io)
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: YOUR_USERNAME/YOUR_REPO
   - Branch: main
   - Main file path: `app.py`
   - Click "Advanced settings"
   - Add secrets:
     ```toml
     GROQ_API_KEY = "gsk_your_actual_key_here"
     ENDEE_URL = "https://your-endee-url.com"
     INDEX_NAME = "RAG"
     ```
   - Click "Deploy"

4. **Initialize the Index**
   - Once deployed, open your app
   - Click "Initialize Index" in sidebar
   - Upload sample.txt
   - Click "Ingest Document"
   - Test with questions!

## Option 2: Deploy Endee on Railway

Railway is free and easy for deploying Endee:

1. **Create Railway Account**: [railway.app](https://railway.app)

2. **Create New Project** → Deploy from GitHub

3. **Add Dockerfile** to your Endee repo (if not exists)

4. **Set Environment Variables**:
   ```
   NDD_DATA_DIR=/data
   PORT=8080
   ```

5. **Deploy** and get the public URL

6. **Use this URL** in your Streamlit app's secrets

## Option 3: Local Demo with ngrok

For quick testing/demo:

1. **Start Endee locally**:
   ```bash
   cd /mnt/d/endee-prj
   ./run.sh
   ```

2. **Start ngrok**:
   ```bash
   ngrok http 8080
   ```

3. **Copy the https URL** (e.g., `https://abc123.ngrok-free.app`)

4. **Update .env**:
   ```
   ENDEE_URL=https://abc123.ngrok-free.app
   ```

5. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```

6. **Share the Streamlit URL** with others

**Note**: ngrok free tier URLs expire after 2 hours and change each time.

## Option 4: Full Docker Deployment

Deploy both Endee and Streamlit together:

1. **Create docker-compose.yml**:
   ```yaml
   version: '3.8'
   services:
     endee:
       image: endeeio/endee-server:latest
       ports:
         - "8080:8080"
       volumes:
         - endee-data:/data
     
     streamlit:
       build: .
       ports:
         - "8501:8501"
       environment:
         - ENDEE_URL=http://endee:8080
         - GROQ_API_KEY=${GROQ_API_KEY}
       depends_on:
         - endee
   
   volumes:
     endee-data:
   ```

2. **Deploy to any cloud provider** that supports Docker Compose

## Troubleshooting

### Endee Connection Issues
- Make sure Endee URL is publicly accessible
- Check firewall settings
- Verify Endee is running: `curl YOUR_ENDEE_URL/api/v1/health`

### Streamlit Secrets Not Working
- Secrets format must be TOML
- No quotes around keys
- Restart app after changing secrets

### Model Errors
- Make sure Groq API key is valid
- Check Groq API limits
- Try a different model if one is deprecated

## Cost Considerations

- **Streamlit Cloud**: Free tier available (1 app)
- **Groq API**: Free tier with rate limits
- **Endee Hosting**: 
  - Railway: Free tier available
  - Render: Free tier available
  - AWS/GCP: Pay as you go

## Security Notes

- Never commit `.env` file to GitHub
- Use Streamlit secrets for production
- Enable authentication if deploying publicly
- Consider rate limiting for public demos
