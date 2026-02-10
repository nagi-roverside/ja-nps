# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Account** - Your code must be in a GitHub repository
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **OpenAI API Key** - Get one from [platform.openai.com](https://platform.openai.com)

## Deployment Steps

### 1. Push Code to GitHub
```bash
git init
git add .
git commit -m "Add NPS Growth Engine with login system"
git branch -M main
git remote add origin https://github.com/yourusername/nps-growth-engine.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository and branch
4. Set the main file path to `nps_growth_engine_internal.py`
5. Click "Deploy"

### 3. Configure Secrets on Streamlit Cloud

After deployment, configure secrets:

1. Go to your app settings on Streamlit Cloud
2. Click "Secrets" in the sidebar
3. Add your OpenAI API key:
```toml
OPENAI_API_KEY = "your-actual-api-key-here"
```

### 4. Environment Variables (Optional)

If you prefer environment variables, add in Streamlit Cloud settings:
```
OPENAI_API_KEY=your-actual-api-key-here
```

## File Structure for Deployment

Your repository should contain:
```
├── nps_growth_engine_internal.py    # Main application
├── requirements.txt                  # Dependencies
├── .streamlit/
│   └── secrets.toml                 # Local secrets (optional)
├── README.md                        # Documentation
└── DEPLOYMENT.md                    # This file
```

## Required Dependencies

The `requirements.txt` file includes:
- `streamlit` - Web framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `plotly` - Interactive charts
- `vaderSentiment` - Sentiment analysis
- `wordcloud` - Word cloud generation
- `matplotlib` - Plotting
- `openai` - OpenAI API client
- `requests` - HTTP requests
- `python-dotenv` - Environment variables

## Demo Users

The app includes three demo users:
- **admin** / **Admin@123!** - Full access
- **analyst** / **Analyst@456!** - Analysis access
- **viewer** / **Viewer@789!** - View-only access

## Data Sources

1. **Google Drive Download**: Pre-configured URL for sample data
2. **CSV Upload**: Upload your own NPS data

## Troubleshooting Common Issues

### 1. Module Not Found Errors
- Ensure all dependencies are in `requirements.txt`
- Streamlit Cloud installs packages automatically

### 2. OpenAI API Key Issues
- Verify API key is correct and has credits
- Check secrets configuration on Streamlit Cloud
- Ensure key is added to `.streamlit/secrets.toml` or environment variables

### 3. Data Download Failures
- Check internet connectivity
- Verify Google Drive URL is accessible
- Use CSV upload as alternative

### 4. Login Problems
- Demo users are created automatically on first run
- Credentials stored in `user_credentials.json`
- Passwords are hashed for security

## Security Notes for Production

1. **Change Demo Passwords** - Update passwords in production
2. **Use Environment Variables** - For sensitive data
3. **Implement User Management** - Add registration/reset for production
4. **Rate Limiting** - Consider adding for API calls
5. **Data Encryption** - For sensitive NPS data

## Performance Optimization

1. **Caching**: The app uses Streamlit caching for embeddings
2. **Batch Processing**: Embeddings generated in batches
3. **Lazy Loading**: Data loaded only when needed
4. **Session State**: User data persists across interactions

## Monitoring

1. **Streamlit Cloud Logs**: Check app logs for errors
2. **OpenAI Usage**: Monitor API usage and costs
3. **User Analytics**: Consider adding analytics for usage patterns

## Support

For issues with deployment:
1. Check Streamlit Cloud documentation
2. Review error logs in app settings
3. Test locally before deploying
4. Ensure all file paths are correct