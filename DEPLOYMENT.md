# Deployment Guide - Smart Notes

## 🚀 Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://share.streamlit.io/)
- OpenAI API key

---

## Step 1: Push to GitHub

### Initialize Git Repository (if not already done)

```powershell
# Navigate to project directory
cd D:\dev\ai\projects\Smart-Notes

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Smart Notes app with OCR and AI reasoning"
```

### Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `smart-notes`
3. Description: "AI-powered study notes generator with OCR, audio transcription, and multi-stage reasoning"
4. Choose **Public** or **Private**
5. **Do NOT** initialize with README (we already have one)
6. Click "Create repository"

### Push to GitHub

```powershell
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/smart-notes.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy to Streamlit Cloud

### 1. Sign in to Streamlit Cloud
- Go to https://share.streamlit.io/
- Sign in with GitHub

### 2. Create New App
1. Click **"New app"**
2. Choose your repository: `YOUR_USERNAME/smart-notes`
3. Branch: `main`
4. Main file path: `app.py`
5. App URL: Choose a custom subdomain (e.g., `smart-notes-ai`)

### 3. Configure Secrets (Important!)
Before deploying, add your OpenAI API key:

1. Click **"Advanced settings"**
2. Go to **"Secrets"** section
3. Add this TOML format:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

4. Click **"Save"**

### 4. Deploy
- Click **"Deploy!"**
- Wait 3-5 minutes for deployment
- Your app will be live at: `https://YOUR-SUBDOMAIN.streamlit.app`

---

## Step 3: Verify Deployment

### Test the following:
- ✅ App loads without errors
- ✅ Can paste notes and generate study notes
- ✅ Can upload images (OCR works)
- ✅ Can export to JSON
- ✅ Sessions are saved

---

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue: EasyOCR fails to install
**Solution**: System dependencies are in `packages.txt` (already included)

### Issue: "OpenAI API key not found"
**Solution**: Add API key to Streamlit Secrets (see Step 2.3)

### Issue: App crashes on startup
**Solutions**:
1. Check Streamlit Cloud logs (bottom of deploy page)
2. Verify all imports work locally first
3. Check if any file paths are hardcoded (use relative paths)

### Issue: OCR cache not working
**Solution**: This is normal on Streamlit Cloud - cache is ephemeral and resets on redeploy

---

## 📊 Monitoring Your App

### View Logs
- Go to your app dashboard
- Click **"Manage app"** → **"Logs"**
- See real-time errors and warnings

### Usage Analytics
- Check **"Analytics"** tab for:
  - Daily active users
  - App load time
  - Error rates

### Update App
```powershell
# Make changes locally
git add .
git commit -m "Description of changes"
git push origin main

# Streamlit Cloud auto-deploys on push!
```

---

## 🔒 Security Best Practices

### Never Commit:
- ✅ `.env` files (already in .gitignore)
- ✅ API keys in code
- ✅ User data or sessions
- ✅ Cache files

### Always Use:
- ✅ Streamlit Secrets for API keys
- ✅ Environment variables
- ✅ `.gitignore` for sensitive files

---

## 💰 Cost Considerations

### Streamlit Cloud (Free Tier)
- ✅ 1 private app OR 3 public apps
- ✅ 1 GB memory
- ✅ 1 CPU core
- ✅ Unlimited runtime
- ✅ Community support

### Upgrade to Paid ($20/month) if you need:
- More private apps
- Higher resource limits (memory/CPU)
- Custom domains
- Priority support

### OpenAI API Costs
- Average cost per session: $0.05-0.15
- 1000 sessions = ~$50-150
- **Tip**: Use local LLM (Ollama) to reduce costs!

---

## 🎯 Next Steps

After successful deployment:

1. **Share your app**: Copy the URL and share with users
2. **Monitor usage**: Check analytics weekly
3. **Iterate**: Based on user feedback
4. **Scale**: Upgrade if needed

### Optional Enhancements:
- Add custom domain (Streamlit paid tier)
- Implement analytics tracking
- Add user feedback form
- Set up error alerts (Sentry)

---

## 📝 Quick Reference

### App URL Structure
```
https://YOUR-SUBDOMAIN.streamlit.app
```

### GitHub Repository
```
https://github.com/YOUR_USERNAME/smart-notes
```

### Update Secrets (Anytime)
1. Streamlit Cloud Dashboard
2. Your App → "Settings" → "Secrets"
3. Edit TOML format
4. Click "Save" (app auto-restarts)

---

## ⚡ Quick Deploy Commands

```powershell
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/smart-notes.git
git push -u origin main

# Future updates
git add .
git commit -m "Your update message"
git push
```

---

## 🆘 Support

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: https://github.com/YOUR_USERNAME/smart-notes/issues

---

**Last Updated**: January 31, 2026
**Deployment Status**: Ready ✅
