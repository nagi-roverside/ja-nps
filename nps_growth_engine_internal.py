import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
import requests
import hashlib
import json
import os
from datetime import datetime
import re

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
    st.session_state.data_loaded = False

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
# DATA DOWNLOAD FROM GOOGLE DRIVE (AUTOMATIC)
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
# AUTOMATIC DATA LOADING
# --------------------------------------------------

if st.session_state.df is None and not st.session_state.data_loaded:
    with st.spinner("📥 Loading NPS data from Google Drive..."):
        df = download_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Failed to load data. Please check your internet connection.")
            st.stop()

# --------------------------------------------------
# MAIN APP (AFTER LOGIN)
# --------------------------------------------------

st.title(f"🚀 Internal NPS Intelligence & Growth Engine")
st.markdown(f"Welcome, **{st.session_state.username}** ({st.session_state.role})")

# --------------------------------------------------
# DATA PROCESSING
# --------------------------------------------------

df = st.session_state.df

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
    # TABS (3 TABS NOW)
    # --------------------------------------------------
    
    tab1, tab2, tab3 = st.tabs(["📊 NPS Dashboard", "💬 AI Chat Assistant", "📋 Data Table"])
    
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
    # TAB 2 — AI CHAT ASSISTANT (ENHANCED)
    # ==================================================
    
    with tab2:
        
        st.subheader("💬 AI Chat Assistant")
        st.markdown("Ask questions about your NPS data. I can create charts and visualizations for you!")
        
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
        # Chat Interface with History Display
        # ---------------------------
        
        # Display chat history in a scrollable container
        st.subheader("Conversation History")
        
        chat_container = st.container(height=400, border=True)
        
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    # Check if message contains chart instructions
                    content = message["content"]
                    
                    # Check for plotly chart patterns
                    if "```plotly" in content or "chart:" in content.lower() or "plot:" in content.lower():
                        # Try to extract and render chart
                        try:
                            # Simple pattern matching for chart descriptions
                            if "bar chart" in content.lower() or "histogram" in content.lower():
                                # Create a simple bar chart based on rating distribution
                                rating_counts = df["rating"].value_counts().sort_index()
                                fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                                           title="Rating Distribution",
                                           labels={"x": "Rating", "y": "Count"})
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif "pie chart" in content.lower() or "nps category" in content.lower():
                                # Create pie chart of NPS categories
                                category_counts = df["nps_category"].value_counts()
                                fig = px.pie(values=category_counts.values, names=category_counts.index,
                                           title="NPS Category Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif "sentiment" in content.lower():
                                # Create sentiment distribution chart
                                fig = px.histogram(df, x="sentiment", nbins=20,
                                                 title="Sentiment Distribution",
                                                 labels={"sentiment": "Sentiment Score"})
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif "scatter" in content.lower() or "correlation" in content.lower():
                                # Create scatter plot of rating vs sentiment
                                fig = px.scatter(df, x="rating", y="sentiment",
                                               color="nps_category",
                                               title="Rating vs Sentiment by NPS Category",
                                               labels={"rating": "Rating", "sentiment": "Sentiment Score"})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Display the text content as well
                            st.markdown(content)
                            
                        except Exception as e:
                            st.markdown(content)
                            st.warning(f"Chart rendering attempted but failed: {str(e)}")
                    else:
                        st.markdown(content)
        
        # ---------------------------
        # Chat Input at Bottom (Multiline)
        # ---------------------------
        
        st.divider()
        st.subheader("Your Message")
        
        # Create a multiline text input at the bottom
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Type your message here...",
                height=100,
                placeholder="Ask me anything about the NPS data. You can ask for charts like 'Show me a bar chart of ratings' or 'Create a pie chart of NPS categories'",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
            clear_button = st.button("Clear Chat", type="secondary", use_container_width=True)
        
        # Handle clear button
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Handle send button
        if send_button and user_input.strip():
            prompt = user_input.strip()
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if API key is available
            if not st.session_state.api_key:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I need an OpenAI API key to answer your questions. Please provide one in the API key field above.",
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()
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
                
                # Prepare system prompt with conversation history and chart capabilities
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

Available data columns: date, rating, text, nps_score, nps_category, sentiment

{context}

IMPORTANT: When the user asks for charts or visualizations, you can create them by including special markers:
- For a bar chart of ratings: include "CHART:bar:rating" in your response
- For a pie chart of NPS categories: include "CHART:pie:nps_category"  
- For a histogram of sentiment: include "CHART:histogram:sentiment"
- For a scatter plot of rating vs sentiment: include "CHART:scatter:rating:sentiment"

Use the data above to answer the user's question. If you don't have enough information, ask clarifying questions.
Be conversational and helpful."""
                
                # Generate response
                with st.spinner("Thinking..."):
                    try:
                        client = OpenAI(api_key=st.session_state.api_key)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7
                        )
                        
                        answer = response.choices[0].message.content
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I encountered an error: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })
                        st.rerun()
    
    # ==================================================
    # TAB 3 — DATA TABLE VIEW
    # ==================================================
    
    with tab3:
        
        st.subheader("📋 Raw Data Table")
        st.markdown(f"Showing {len(df)} reviews. Use the filters below to explore the data.")
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Columns", len(df.columns))
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Rating filter
            min_rating, max_rating = st.slider(
                "Rating Range",
                min_value=1,
                max_value=5,
                value=(1, 5)
            )
        
        with col2:
            # NPS Category filter
            categories = ["All"] + df["nps_category"].unique().tolist()
            selected_category = st.selectbox(
                "NPS Category",
                categories
            )
        
        with col3:
            # Date range filter
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Rating filter
        filtered_df = filtered_df[(filtered_df["rating"] >= min_rating) & (filtered_df["rating"] <= max_rating)]
        
        # NPS Category filter
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["nps_category"] == selected_category]
        
        # Date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["date"].dt.date >= start_date) & 
                (filtered_df["date"].dt.date <= end_date)
            ]
        
        st.divider()
        
        # Display filtered data
        st.subheader(f"📊 Filtered Data ({len(filtered_df)} reviews)")
        
        # Show data table with pagination
        page_size = 20
        total_pages = max(1, len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0))
        
        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
        
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        # Display the current page
        st.dataframe(
            filtered_df.iloc[start_idx:end_idx][["date", "rating", "nps_category", "sentiment", "text"]],
            use_container_width=True,
            height=400
        )
        
        st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {len(filtered_df)}")
        
        # Download option
        st.divider()
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered CSV",
                data=csv,
                file_name=f"nps_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            csv_all = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Data CSV",
                data=csv_all,
                file_name=f"nps_all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Data statistics
        st.divider()
        st.subheader("📈 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Rating", round(filtered_df["rating"].mean(), 2))
            st.metric("Median Rating", filtered_df["rating"].median())
        
        with col2:
            st.metric("Average Sentiment", round(filtered_df["sentiment"].mean(), 3))
            st.metric("Positive Reviews", len(filtered_df[filtered_df["sentiment"] > 0.05]))
        
        with col3:
            category_counts = filtered_df["nps_category"].value_counts()
            for category, count in category_counts.items():
                st.metric(f"{category}s", count)

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
    st.session_state.data_loaded = False
    st.rerun()
