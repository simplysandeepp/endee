# Quick Deployment Steps

## 🚀 Deploy Your RAG App in 5 Minutes

### Step 1: Push to GitHub

```bash
cd project-RAG

# Initialize git
git init

# Add all files (except .env - it's in .gitignore)
git add .

# Commit
git commit -m "RAG application with Endee and Groq"

# Create a new repository on GitHub.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Make Endee Publicly Accessible

**Quick Option (Testing): Use ngrok**

```bash
# In WSL terminal where Endee is running
ngrok http 8080
```

Copy the https URL (e.g., `https://abc-123-def.ngrok-free.app`)

**Production Option: Deploy Endee on Railway**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. New Project → Deploy from GitHub → Select endee repo
4. Add environment variable: `PORT=8080`
5. Copy the public URL

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository**: YOUR_USERNAME/YOUR_REPO_NAME
   - **Branch**: main
   - **Main file path**: `app.py`
5. Click **"Advanced settings"**
6. In the **Secrets** box, paste:
   ```toml
   GROQ_API_KEY = "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"
   ENDEE_URL = "YOUR_ENDEE_PUBLIC_URL"
   INDEX_NAME = "RAG"
   ```
7. Click **"Deploy"**

### Step 4: Initialize and Test

1. Wait for deployment to complete (2-3 minutes)
2. Open your app URL (e.g., `https://your-app.streamlit.app`)
3. Click **"Initialize Index"** in sidebar
4. Upload `sample.txt` from the data folder
5. Click **"Ingest Document"**
6. Ask: "Who was the engineer living near the banyan tree?"
7. Get answer: "Professor Devendra Rao"

### Step 5: Share Your Demo

Your app is now live! Share the URL:
- `https://your-app-name.streamlit.app`

## 📝 Notes

- **ngrok URLs expire** after 2 hours on free tier
- For permanent demo, deploy Endee on Railway/Render
- Streamlit Cloud free tier: 1 app, community support
- Keep your Groq API key secret!

## 🎉 You're Done!

Your RAG application is now live and accessible to anyone with the link!
