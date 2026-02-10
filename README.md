# NPS Growth Engine - Enhanced Version

## Overview
An enhanced Streamlit application for analyzing Net Promoter Score (NPS) data with AI-powered insights, login system, and real-time data download capabilities.

## New Features Implemented

### 1. **Login System**
- Secure authentication with demo users
- Credentials stored in `user_credentials.json`
- Three demo users with different roles:
  - `admin` / `Admin@123!` (admin role)
  - `analyst` / `Analyst@456!` (analyst role)
  - `viewer` / `Viewer@789!` (viewer role)

### 2. **Real-time Data Download**
- Download NPS data directly from Google Drive
- URL: `https://drive.google.com/uc?export=download&id=1pYGLkoNPUe2j-K9x41kKw4p1Opgr03Ls`
- Alternative: Upload CSV files manually

### 3. **Improved Dashboard**
- Removed time series chart (data is only for one month)
- Added data summary with date range information
- Rating distribution visualization
- Enhanced executive summary

### 4. **AI Chat Assistant**
- Conversational chatbot interface
- Maintains conversation history
- Uses embeddings for semantic search
- Leverages previous messages for context
- Clear chat history functionality

### 5. **OpenAI API Key Management**
- Store API key in Streamlit secrets (`.streamlit/secrets.toml`)
- Fallback to manual input if secrets not configured
- Automatic embedding generation on app start

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
   - Option A: Add to `.streamlit/secrets.toml`:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - Option B: Enter manually in the app

## Usage

1. Run the Streamlit app:
```bash
streamlit run nps_growth_engine_internal.py
```

2. Login with demo credentials:
   - Username: `admin`, Password: `Admin@123!`
   - Username: `analyst`, Password: `Analyst@456!`
   - Username: `viewer`, Password: `Viewer@789!`

3. Choose data source:
   - Download from Google Drive
   - Upload CSV file

4. Explore the dashboard and chat with AI assistant

## File Structure

- `nps_growth_engine_internal.py` - Main application
- `user_credentials.json` - User credentials (auto-generated)
- `.streamlit/secrets.toml` - API key storage
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Data Requirements

CSV files must contain these columns:
- `date` - Review date
- `rating` - Rating (1-5 scale)
- `text` - Review text

## Security Notes

- Passwords are hashed using SHA-256
- API keys are stored securely in Streamlit secrets
- Demo users have strong passwords
- Logout functionality clears session data

## Troubleshooting

1. **OpenAI API errors**: Ensure API key is valid and has sufficient credits
2. **Data download issues**: Check internet connection and URL accessibility
3. **Login problems**: Verify credentials in `user_credentials.json`
4. **Missing dependencies**: Run `pip install -r requirements.txt`