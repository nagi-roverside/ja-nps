# NPS Growth Engine - Production Version

## Overview
A production-ready Streamlit application for analyzing Net Promoter Score (NPS) data with AI-powered insights, secure login system, and real-time data download capabilities.

## Key Features

### 1. **Secure Login System**
- Authentication using Streamlit secrets (no hardcoded users)
- Password hashing with SHA-256
- Role-based access control (admin, analyst, viewer)
- Session management and logout functionality

### 2. **Automatic Data Loading**
- Data loads automatically from Google Drive on app start
- No manual intervention required
- URL: `https://drive.google.com/uc?export=download&id=1pYGLkoNPUe2j-K9x41kKw4p1Opgr03Ls`

### 3. **Three-Tab Interface**
- **📊 NPS Dashboard**: Analytics and visualizations
- **💬 AI Chat Assistant**: Conversational chatbot with chart capabilities
- **📋 Data Table**: Interactive data exploration with filtering

### 4. **AI Chat Assistant with Chart Generation**
- Conversational interface with history
- Real-time Plotly chart generation
- Can create bar charts, pie charts, histograms, and scatter plots
- Multiline text input at bottom of interface

### 5. **Data Table with Advanced Features**
- Interactive filtering by rating, category, and date
- Pagination for large datasets
- Export filtered or all data as CSV
- Quick statistics for filtered data

## Streamlit Cloud Deployment

### 1. **Repository Setup**
- Repository: `https://github.com/nagi-roverside/ja-nps`
- Main file: `nps_growth_engine_internal.py`

### 2. **Secrets Configuration**
Add these secrets in Streamlit Cloud:

```toml
# OpenAI API Key
OPENAI_API_KEY = "your-openai-api-key-here"

# User Credentials (format: username:password:role,username2:password2:role2)
USERS = "admin:AdminPass123:admin,analyst:AnalystPass456:analyst,viewer:ViewerPass789:viewer"
```

### 3. **Deployment Steps**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select repository: `nagi-roverside/ja-nps`
4. Set main file path: `nps_growth_engine_internal.py`
5. Configure secrets as shown above
6. Click "Deploy"

## Local Development

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Configure Secrets**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
USERS = "admin:AdminPass123:admin,analyst:AnalystPass456:analyst"
```

### 3. **Run the Application**
```bash
streamlit run nps_growth_engine_internal.py
```

## File Structure

```
ja-nps/
├── nps_growth_engine_internal.py    # Main application
├── requirements.txt                  # Python dependencies
├── README.md                         # This documentation
├── DEPLOYMENT.md                     # Detailed deployment guide
├── .gitignore                        # Git exclusion rules
└── .streamlit/
    ├── .gitkeep                      # Directory structure
    └── secrets.toml                  # Local secrets (optional)
```

## Data Requirements

CSV files must contain these columns:
- `date` - Review date (YYYY-MM-DD format)
- `rating` - Rating (1-5 scale)
- `text` - Review text content

## Security Features

- **No hardcoded credentials** - All users configured via secrets
- **Password hashing** - SHA-256 for secure password storage
- **Session management** - Automatic logout and session cleanup
- **API key security** - OpenAI keys stored in Streamlit secrets
- **Input validation** - All user inputs are validated

## Chatbot Chart Capabilities

The AI chatbot can create these visualizations:
- **Bar charts**: Rating distribution, keyword mentions
- **Pie charts**: NPS category distribution
- **Histograms**: Sentiment score distribution
- **Scatter plots**: Rating vs sentiment correlation

Users can request charts with natural language:
- "Show me a bar chart of ratings"
- "Create a pie chart of NPS categories"
- "Plot sentiment distribution as a histogram"

## Troubleshooting

### 1. **Login Issues**
- Verify user credentials in Streamlit secrets
- Check format: `username:password:role`
- Ensure passwords don't contain special characters that break parsing

### 2. **OpenAI API Errors**
- Verify API key is valid and has credits
- Check rate limits and usage quotas
- Ensure key is added to Streamlit secrets

### 3. **Data Loading Problems**
- Check internet connectivity
- Verify Google Drive URL is accessible
- Ensure CSV has required columns: date, rating, text

### 4. **Chart Generation Failures**
- Ensure OpenAI API key is configured
- Check if embeddings have been generated
- Verify data is loaded successfully

## Performance Optimization

- **Caching**: Data download and embeddings are cached
- **Batch processing**: Embeddings generated in batches of 100
- **Lazy loading**: Charts generated only when requested
- **Session state**: User data persists across interactions

## Support

For deployment issues:
1. Check Streamlit Cloud logs
2. Verify secrets configuration
3. Test locally before deploying
4. Ensure all dependencies are in requirements.txt

## License

This application is ready for production use with proper security configuration.