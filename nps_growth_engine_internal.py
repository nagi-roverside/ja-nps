import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
import requests
import hashlib
import json
import os
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(page_title="NPS Growth Engine", layout="wide")

# --------------------------------------------------
# LOGIN SYSTEM
# --------------------------------------------------

CREDENTIALS_FILE = "user_credentials.json"

def load_credentials():
    """Load user credentials from file"""
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_credentials(credentials):
    """Save user credentials to file"""
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f, indent=2)

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_demo_users():
    """Initialize demo users if credentials file doesn't exist"""
    credentials = load_credentials()
    if not credentials:
        demo_users = {
            "admin": {
                "password_hash": hash_password("Admin@123!"),
                "role": "admin"
            },
            "analyst": {
                "password_hash": hash_password("Analyst@456!"),
                "role": "analyst"
            },
            "viewer": {
                "password_hash": hash_password("Viewer@789!"),
                "role": "viewer"
            }
        }
        save_credentials(demo_users)
        return demo_users
    return credentials

# Initialize demo users
credentials = init_demo_users()

# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def check_login(username, password):
    """Check if username and password are valid"""
    if username in credentials:
        stored_hash = credentials[username]["password_hash"]
        return hash_password(password) == stored_hash
    return False

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.chat_history = []
    st.session_state.df = None
    st.session_state.review_embeddings = None
    st.session_state.api_key = None

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

if not st.session_state.authenticated:
    st.title("🔐 NPS Growth Engine - Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", type="primary", use_container_width=True):
                if check_login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = credentials[username]["role"]
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            
            st.divider()
            st.markdown("### Demo Users")
            st.markdown("""
            - **admin** / Admin@123!
            - **analyst** / Analyst@456!
            - **viewer** / Viewer@789!
            """)
    
    st.stop()

# --------------------------------------------------
# MAIN APP (AFTER LOGIN)
# --------------------------------------------------

st.title(f"🚀 Internal NPS Intelligence & Growth Engine")
st.markdown(f"Welcome, **{st.session_state.username}** ({st.session_state.role}) | [Logout](#)")

# --------------------------------------------------
# DATA DOWNLOAD FROM GOOGLE DRIVE
# --------------------------------------------------

DATA_URL = "https://drive.google.com/uc?export=download&id=1pYGLkoNPUe2j-K9x41kKw4p1Opgr03Ls"

@st.cache_data(show_spinner=True)
def download_data():
    """Download data from Google Drive URL"""
    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with open("temp_reviews.csv", "wb") as f:
            f.write(response.content)
        
        # Load into DataFrame
        df = pd.read_csv("temp_reviews.csv")
        
        # Clean up
        os.remove("temp_reviews.csv")
        
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

# --------------------------------------------------
# DATA LOADING SECTION
# --------------------------------------------------

st.sidebar.header("📊 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Download from Google Drive", "Upload CSV File"]
)

df = None

if data_source == "Download from Google Drive":
    if st.sidebar.button("Download Data", type="primary"):
        with st.spinner("Downloading data from Google Drive..."):
            df = download_data()
            if df is not None:
                st.session_state.df = df
                st.success(f"Downloaded {len(df)} reviews successfully!")
                st.rerun()
    
    if st.session_state.df is not None:
        df = st.session_state.df
else:
    uploaded_file = st.sidebar.file_uploader("Upload Review CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

# --------------------------------------------------
# MAIN CONTENT (IF DATA LOADED)
# --------------------------------------------------

if df is not None:
    # Validate required columns
    required_cols = ["date", "rating", "text"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain: date, rating, text columns")
        st.stop()
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    # --------------------------------------------------
    # NPS CALCULATION
    # --------------------------------------------------
    
    df["nps_score"] = df["rating"] * 2
    
    def classify(score):
        if score >= 9:
            return "Promoter"
        elif score >= 7:
            return "Passive"
        else:
            return "Detractor"
    
    df["nps_category"] = df["nps_score"].apply(classify)
    
    promoters = len(df[df["nps_category"] == "Promoter"])
    detractors = len(df[df["nps_category"] == "Detractor"])
    passives = len(df[df["nps_category"] == "Passive"])
    total = len(df)
    
    nps = ((promoters - detractors) / total) * 100
    
    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    
    tab1, tab2 = st.tabs(["📊 NPS Dashboard", "💬 AI Chat Assistant"])
    
    # ==================================================
    # TAB 1 — DASHBOARD
    # ==================================================
    
    with tab1:
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall NPS", round(nps, 2))
        col2.metric("Promoters", promoters)
        col3.metric("Passives", passives)
        col4.metric("Detractors", detractors)
        
        st.divider()
        
        # ---------------------------
        # Data Summary (instead of time series)
        # ---------------------------
        
        st.subheader("📋 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Date Range**")
            min_date = df["date"].min().strftime("%Y-%m-%d")
            max_date = df["date"].max().strftime("%Y-%m-%d")
            st.metric("From", min_date)
            st.metric("To", max_date)
            
            # Check if data is only for a month
            date_range = (df["date"].max() - df["date"].min()).days
            if date_range <= 31:
                st.info("📅 Data covers approximately one month. Time series analysis may not be meaningful.")
        
        with col2:
            st.markdown("**Rating Distribution**")
            rating_counts = df["rating"].value_counts().sort_index()
            st.bar_chart(rating_counts)
        
        # ---------------------------
        # Keyword Segmentation
        # ---------------------------
        
        st.subheader("🔎 High-Risk Keyword Detection")
        
        keywords = ["scam", "refund", "billing", "charge", "lawyer",
                    "delay", "helpful", "cancel", "support"]
        
        keyword_data = {}
        
        for word in keywords:
            keyword_data[word] = df["text"].str.contains(
                word, case=False, na=False
            ).sum()
        
        keyword_df = pd.DataFrame.from_dict(
            keyword_data, orient="index", columns=["Mentions"]
        )
        
        st.bar_chart(keyword_df)
        
        # ---------------------------
        # Sentiment Analysis
        # ---------------------------
        
        st.subheader("🧠 Sentiment Comparison")
        
        analyzer = SentimentIntensityAnalyzer()
        
        df["sentiment"] = df["text"].apply(
            lambda x: analyzer.polarity_scores(str(x))["compound"]
        )
        
        promoter_sent = df[df["nps_category"] == "Promoter"]["sentiment"].mean()
        detractor_sent = df[df["nps_category"] == "Detractor"]["sentiment"].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Avg Promoter Sentiment", round(promoter_sent, 3))
        col2.metric("Avg Detractor Sentiment", round(detractor_sent, 3))
        
        # ---------------------------
        # Word Cloud
        # ---------------------------
        
        st.subheader("🔥 Detractor Word Cloud")
        
        detractor_text = " ".join(
            df[df["nps_category"] == "Detractor"]["text"].astype(str)
        )
        
        if detractor_text.strip():
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(detractor_text)
            
            fig_wc, ax = plt.subplots()
            ax.imshow(wordcloud)
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No detractor reviews available for word cloud.")
        
        # ---------------------------
        # Executive Summary
        # ---------------------------
        
        st.subheader("📊 Executive Summary")
        
        if nps < 0:
            health = "Critical Reputation Risk"
        elif nps < 30:
            health = "Needs Improvement"
        else:
            health = "Strong Advocacy Engine"
        
        st.markdown(f"""
        ### Business Health: {health}
        
        **Current NPS:** {round(nps,2)}
        
        **Key Risks Identified:**
        - High detractor volume
        - Negative sentiment intensity gap
        - Friction keywords indicate operational issues
        
        **Immediate Actions:**
        1. Audit billing/refund flow
        2. Implement 24-hour detractor callback loop
        3. Publicly showcase promoter testimonials
        4. Trigger post-resolution review capture
        
        **Strategic Goal:**
        Convert promoters into referral advocates to build
        sustainable word-of-mouth growth.
        """)
    
    # ==================================================
    # TAB 2 — AI CHAT ASSISTANT (CONVERSATIONAL)
    # ==================================================
    
    with tab2:
        
        st.subheader("💬 AI Chat Assistant")
        st.markdown("Ask questions about your NPS data in a conversational way.")
        
        # ---------------------------
        # OpenAI API Key from Secrets
        # ---------------------------
        
        # Try to get API key from Streamlit secrets
        try:
            if 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
                st.session_state.api_key = api_key
                st.success("✅ OpenAI API key loaded from secrets")
            else:
                st.warning("OpenAI API key not found in secrets. Please add it to `.streamlit/secrets.toml`")
                api_key = st.text_input("OpenAI API Key", type="password", 
                                       help="Enter your OpenAI API key or add it to secrets.toml")
                if api_key:
                    st.session_state.api_key = api_key
        except:
            # If secrets not configured, use text input
            api_key = st.text_input("OpenAI API Key", type="password",
                                   help="Enter your OpenAI API key")
            if api_key:
                st.session_state.api_key = api_key
        
        # ---------------------------
        # Generate Embeddings on App Start
        # ---------------------------
        
        @st.cache_data(show_spinner=True)
        def generate_embeddings(texts, api_key):
            """Generate embeddings for review texts"""
            if not api_key:
                return None
            
            client_local = OpenAI(api_key=api_key)
            embeddings = []
            
            # Process in batches for efficiency
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = client_local.embeddings.create(
                    model="text-embedding-3-small",
                    input=[str(text) for text in batch]
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
        
        # Generate embeddings if API key is available
        if st.session_state.api_key and st.session_state.review_embeddings is None:
            with st.spinner("Generating embeddings for all reviews (this may take a moment)..."):
                review_texts = df["text"].fillna("").tolist()
                st.session_state.review_embeddings = generate_embeddings(review_texts, st.session_state.api_key)
                st.success(f"Generated embeddings for {len(review_texts)} reviews")
        
        # ---------------------------
        # Chat Interface
        # ---------------------------
        
        # Display chat history
        st.subheader("Conversation")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your NPS data..."):
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if API key is available
            if not st.session_state.api_key:
                with st.chat_message("assistant"):
                    st.error("Please provide an OpenAI API key to use the chat assistant.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I need an OpenAI API key to answer your questions. Please provide one in the API key field above.",
                        "timestamp": datetime.now().isoformat()
                    })
            else:
                # Prepare context from embeddings if available
                context = ""
                if st.session_state.review_embeddings is not None:
                    client = OpenAI(api_key=st.session_state.api_key)
                    
                    # Embed the question
                    question_embedding = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=prompt
                    ).data[0].embedding
                    
                    question_embedding = np.array(question_embedding)
                    
                    # Cosine similarity search
                    def cosine_similarity(a, b):
                        return np.dot(a, b) / (
                            np.linalg.norm(a) * np.linalg.norm(b)
                        )
                    
                    similarities = np.array([
                        cosine_similarity(question_embedding, emb)
                        for emb in st.session_state.review_embeddings
                    ])
                    
                    top_k = min(10, len(df))
                    top_indices = similarities.argsort()[-top_k:][::-1]
                    
                    relevant_reviews = df.iloc[top_indices][
                        ["date", "rating", "nps_category", "text"]
                    ]
                    
                    context = f"\nRelevant Reviews:\n{relevant_reviews.to_string()}"
                
                # Prepare system prompt with conversation history
                conversation_history = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content']}" 
                    for msg in st.session_state.chat_history[-6:-1]  # Last 5 messages (excluding current)
                ])
                
                system_prompt = f"""You are a senior customer intelligence strategist analyzing NPS (Net Promoter Score) data.

Previous conversation:
{conversation_history}

Current NPS metrics:
- Overall NPS: {round(nps, 2)}
- Promoters: {promoters}
- Passives: {passives}
- Detractors: {detractors}

{context}

Use the data above to answer the user's question. If you don't have enough information, ask clarifying questions.
Be conversational and helpful."""
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7
                            )
                            
                            answer = response.choices[0].message.content
                            st.markdown(answer)
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"I encountered an error: {str(e)}",
                                "timestamp": datetime.now().isoformat()
                            })
        
        # Add clear chat button
        st.divider()
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# --------------------------------------------------
# NO DATA MESSAGE
# --------------------------------------------------
else:
    st.info("👈 Please load data using the sidebar options to get started.")
    
    # Show data preview if available
    if os.path.exists("reviews.csv"):
        st.subheader("Sample Data Available")
        sample_df = pd.read_csv("reviews.csv", nrows=5)
        st.dataframe(sample_df)
        if st.button("Load Sample Data"):
            df = pd.read_csv("reviews.csv")
            st.session_state.df = df
            st.rerun()

# --------------------------------------------------
# LOGOUT FUNCTIONALITY
# --------------------------------------------------
st.sidebar.divider()
if st.sidebar.button("Logout", type="secondary"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.chat_history = []
    st.session_state.df = None
    st.session_state.review_embeddings = None
    st.session_state.api_key = None
    st.rerun()
