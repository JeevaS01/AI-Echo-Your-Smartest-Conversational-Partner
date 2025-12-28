import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Set page config with white gradient theme
st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    #page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark gradient theme
st.markdown("""
    <style>
    :root {
        --primary-color: #667eea;
        --background-color: #0f1419;
        --secondary-background-color: #1a202c;
    }
    
    body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #16213e 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #e0e7ff;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        padding: 20px;
    }
    
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        padding: 20px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%) !important;
    }
    
    [data-testid="stSidebarNav"] {
        background: transparent !important;
    }
    
    .stContainer {
        background: linear-gradient(135deg, rgba(26,32,44,0.8) 0%, rgba(45,55,72,0.8) 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(26,32,44,0.9) 0%, rgba(45,55,72,0.9) 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .header-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
        padding-top: 50px;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
    }
            /* 1. Hide the default navigation if using multipage apps */
        [data-testid="stSidebarNav"] {display: none;} 

        /* 2. Target only the container of the options, not the main label */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: column;
            gap: 10px;
        }

        /* 3. Style ONLY the clickable option labels (the buttons) */
        /* We use the 'label' that is a direct child of the radiogroup */
       div[data-testid="stRadio"] div[role="radiogroup"] > label {
            background: linear-gradient(135deg, rgba(26,32,44,0.95) 0%,rgba(92, 77, 221, 0.584)  100%); 
            padding: 15px 20px;
            border-radius: 30px;           
            box-shadow: 0 4px 15px rgba(49, 78, 220, 0.344);
            border: 1px solid rgba(0, 0, 0, 0.2);
            color: #e0e7ff;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            display: block;
            font-weight: 500;
        }

        /* 4. Hover effect for buttons */
        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            background-color: #667eea;
            color: white !important;
            transform: translateX(20px);
            border-color: #764ba2;
        }

        /* 5. Style for the SELECTED button */
        /* This ensures only the active choice gets the primary color */
        div[data-testid="stRadio"] div[role="radiogroup"] input:checked + div {
            background-color: #667eea !important;
            color: white !important;
            font-weight: bold;
            border-radius: 10px;
        }
        
        /* 6. Hide the radio circle dot */
        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        /* 7. Ensure the hidden "Select a page" label doesn't take up space or styling */
        div[data-testid="stRadio"] > label {
            display: none !important;
        }
    
    .sentiment-card {
        background: linear-gradient(135deg, rgba(26,32,44,0.95) 0%, rgba(45,55,72,0.95) 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.2);
        color: #e0e7ff;
            
    }
    
    .positive {
        border-left-color: #10b981;
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.05) 100%);
        border: 1px solid rgba(16,185,129,0.2);
            
    }
    
    .negative {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.05) 100%);
        border: 1px solid rgba(239,68,68,0.2);
    }
    
    .neutral {
        border-left-color: #8b5cf6;
        background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(139,92,246,0.2);
    }
    
    /* Sidebar Menu Styling */
    [data-testid="stSidebarNav"] label {
        color: #e0e7ff !important;
        font-weight: 600;
        font-size: 1.1em;
        padding: 15px 20px !important;
        margin: 10px 0 !important;
        border-radius: 10px;
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
    }
    
    [data-testid="stSidebarNav"] label:hover {
        background: linear-gradient(90deg, rgba(102,126,234,0.2) 0%, rgba(102,126,234,0.1) 100%);
        border-left-color: #667eea;
        color: #a5b4fc;
    }
    
    [data-testid="stSidebarNav"] label[aria-selected="true"] {
        background: linear-gradient(90deg, rgba(102,126,234,0.3) 0%, rgba(118,75,162,0.2) 100%);
        border-left-color: #764ba2;
        color: #c7d2fe;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f9ff !important;
    }
    
    p, span, label {
        color: #e0e7ff !important;
    }
    
    /* Tabs styling */
    [data-testid="stTabs"] [role="tab"] {
        background: linear-gradient(135deg, rgba(26,32,44,0.8) 0%, rgba(45,55,72,0.8) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        color: #a0aec0 !important;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
    }
    
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 1px solid #667eea !important;
        color: #ffffff !important;
    }
    
    /* Text area and input styling */
    textarea, input {
        background-color: rgba(30,41,59,0.8) !important;
        color: #e0e7ff !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background-color: rgba(26,32,44,0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Info and Error boxes */
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(45,55,72,0.9) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
    }
            /* Target the specific button container */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            height: 3em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            font-weight: bold;
            font-size: 1.1rem;
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        /* Hover Effect: Glow and Grow */
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(118, 75, 162, 0.5);
            background-color: #667eea;
            color: #ffffff !important;
            

        }

        /* Active/Press Effect: Shrink slightly */
        .stButton > button:active {
            transform: translateY(1px) scale(0.98);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }

        /* Adding a subtle Pulse Animation to draw user attention */
        @keyframes pulse-animation {
            0% { box-shadow: 0 0 0 0px rgba(102, 126, 234, 0.4); }
            100% { box-shadow: 0 0 0 15px rgba(102, 126, 234, 0); }
        }

        .stButton > button {
            animation: pulse-animation 1s infinite;
        }
            /* Add this to your <style> block */
@keyframes shine {
    0% { left: -100%; }
    20% { left: 100%; }
    100% { left: 100%; }
}

.stButton > button {
    position: relative;
    overflow: hidden; /* Important for the shine */
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    transform: skewX(-10deg);
    animation: shine 6s infinite;
}
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / "data" / "cleaned_review.csv",
        base_dir / "cleaned_review.csv",
        Path.cwd() / "cleaned_review.csv",
    ]

    for p in candidate_paths:
        try:
            if p.exists():
                df = pd.read_csv(p)
                df.columns = df.columns.str.strip()
                return df
        except Exception as e:
            st.error(f"Error reading {p}: {e}")

    st.error("cleaned_review.csv not found in 'data/' or project root. Using fallback minimal DataFrame.")
    fallback = pd.DataFrame({
        'location': ['Unknown'],
        'platform': ['Unknown'],
        'rating': [0],
        'sentiment': ['Neutral'],
        'date': [pd.NaT],
        'review_length': [0],
        'processed_reviews': [''],
        'version': ['0.0'],
        'verified_purchase': [False]
    })
    return fallback

data = load_data()

# Initialize sentiment models
@st.cache_resource
def load_models():
    vader_analyzer = SentimentIntensityAnalyzer()
    try:
        from transformers import pipeline as hf_pipeline
        hf_pipeline_model = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.warning(f"Could not load HuggingFace model (will use lightweight fallback): {e}")
        def hf_pipeline_fallback(text):
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return [{'label': 'POSITIVE', 'score': abs(polarity)}]
            elif polarity < -0.1:
                return [{'label': 'NEGATIVE', 'score': abs(polarity)}]
            else:
                return [{'label': 'NEUTRAL', 'score': 1 - abs(polarity)}]
        hf_pipeline_model = hf_pipeline_fallback
    return vader_analyzer, hf_pipeline_model

vader_analyzer, hf_pipeline_model = load_models()

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="header-title">🎯 AI Echo - Your Smartest Conversational Partner</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1em;">Real-time Sentiment Analysis & EDA Dashboard</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0; border-bottom: 2px solid rgba(102,126,234,0.3);">
        <h2 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0;">
            🎯 AI ECHO
        </h2>
        <p style="text-align: center; color: #a0aec0; font-size: 1.2em; margin-top: 5px;">Sentiment Analysis Pro</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <p style="color: #cbd5e0; font-weight: 600; font-size: 1.1em; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">
        📊 Navigation Menu
    </p>
    """, unsafe_allow_html=True)
    
with st.sidebar:
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Real-time Analysis", "📈 EDA Dashboard", "📉 Advanced Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Sidebar Info Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); 
                border: 1px solid rgba(102,126,234,0.3); border-radius: 10px; padding: 15px; margin-top: 30px;">
        <h4 style="color: #a5b4fc; margin-top: 0;">💡 Quick Stats</h4>
        <p style="color: #cbd5e0; font-size: 0.9em; margin: 8px 0;">
            <b>Total Reviews:</b> {}<br>
            <b>Sentiments:</b> 3 Classes<br>
            <b>Models:</b> 3 AI Engines
        </p>
    </div>
    """.format(len(data)), unsafe_allow_html=True)

# Home Page
if page == "🏠 Home":
    st.markdown("### 👋 Welcome to AI Echo Sentiment Analysis Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="sentiment-card">
        <h3 style="color: #a5b4fc;">🎯 Real-time Analysis</h3>
        <p style="color: #cbd5e0;">Get instant sentiment predictions using multiple AI models with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="sentiment-card">
        <h3 style="color: #a5b4fc;">📊 EDA Insights</h3>
        <p style="color: #cbd5e0;">Explore comprehensive exploratory data analysis with visualizations and trends.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="sentiment-card">
        <h3 style="color: #a5b4fc;">📈 Analytics</h3>
        <p style="color: #cbd5e0;">Deep dive into sentiment patterns across locations, platforms, and time periods.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(data):,}")
    
    with col2:
        st.metric("Unique Locations", f"{data['location'].nunique()}")
    
    with col3:
        st.metric("Platforms", f"{data['platform'].nunique()}")
    
    with col4:
        st.metric("Avg Rating", f"{data['rating'].mean():.2f} ⭐")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 💡 Quick Sentiment Overview")
    sentiment_dist = data['sentiment'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="sentiment-card positive">
        <h2 style="color: #10b981; margin-top: 0;">😊 Positive</h2>
        <h1 style="color: #10b981; margin: 10px 0;">{sentiment_dist.get('Positive', 0)}</h1>
        <p style="color: #cbd5e0;">{sentiment_dist.get('Positive', 0)/len(data)*100:.1f}% of reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="sentiment-card neutral">
        <h2 style="color: #8b5cf6; margin-top: 0;">😐 Neutral</h2>
        <h1 style="color: #8b5cf6; margin: 10px 0;">{sentiment_dist.get('Neutral', 0)}</h1>
        <p style="color: #cbd5e0;">{sentiment_dist.get('Neutral', 0)/len(data)*100:.1f}% of reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="sentiment-card negative">
        <h2 style="color: #ef4444; margin-top: 0;">😞 Negative</h2>
        <h1 style="color: #ef4444; margin: 10px 0;">{sentiment_dist.get('Negative', 0)}</h1>
        <p style="color: #cbd5e0;">{sentiment_dist.get('Negative', 0)/len(data)*100:.1f}% of reviews</p>
        </div>
        """, unsafe_allow_html=True)

# Real-time Analysis Page
elif page == "🔍 Real-time Analysis":
    st.markdown("### 🎯 Real-time Sentiment Prediction")
    st.markdown("Enter text below to analyze sentiment using multiple AI models")

    # Model Selection and Input Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            [ "VADER (Recommended)", "TextBlob","BERT"],
            help="Choose the AI model for sentiment analysis"
        )
    
    
    # Input area
    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        user_input = st.text_area(
            "Enter your text here:",
            placeholder="Type a review, comment, or feedback...",
            height=200,
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("""
            <div style="padding: 10px; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border: 1px solid rgba(102,126,234,0.3); border-radius: 10px; margin-bottom: 10px;">
                <h4 style="color: #a5b4fc; margin-top: 0;">🛠️ How to Use</h4>
                <ol style="color: #cbd5e0; font-size: 0.95em; padding-left: 15px; margin: 0;">
                    <li>Type or paste your text into the input area.</li>
                    <li>Select the AI model you want to use for sentiment analysis.</li>
                    <li>Click the "Predict Sentiment" button to see results.</li>
                    <li>View the predicted sentiment along with confidence scores and visualizations.</li>
                    <li>Explore different models to compare results.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Predict Button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_button = st.button("🔍 Predict Sentiment", use_container_width=True, type="primary")

    if len(user_input.strip()) > 10:
        if predict_button and user_input.strip():
            st.markdown("---")
        
            try:
                sentiment_scores = {}
                hf_label = None
                hf_conf = None
                
                if selected_model == "BERT":
                    # Hugging Face BERT Model
                    hf_result = hf_pipeline_model(user_input[:512])[0]
                    hf_label = hf_result['label'].upper()
                    hf_conf = hf_result['score'] * 100
                    
                    # Get all sentiment confidences from HuggingFace model
                    full_results = hf_pipeline_model(user_input[:512])
                    for result in full_results:
                        sentiment = result['label'].upper()
                        score = result['score']
                        sentiment_scores[sentiment] = score * 100
                    
                    # Ensure all sentiments are present
                    for sent in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                        if sent not in sentiment_scores:
                            sentiment_scores[sent] = 0
                
                elif selected_model == "VADER (Recommended)":
                    # VADER Sentiment Analysis
                    vader_scores = vader_analyzer.polarity_scores(user_input)
                    neg = vader_scores['neg'] * 100
                    neu = vader_scores['neu'] * 100
                    pos = vader_scores['pos'] * 100
                    
                    sentiment_scores = {
                        'POSITIVE': pos,
                        'NEGATIVE': neg,
                        'NEUTRAL': neu
                    }
                    
                    # Determine primary sentiment
                    if pos >= neg and pos >= neu:
                        hf_label = 'POSITIVE'
                        hf_conf = pos
                    
                    elif neg >= pos and neg >= neu:
                        hf_label = 'NEGATIVE'
                        hf_conf = neg
                    else:
                        hf_label = 'NEUTRAL'
                        hf_conf = neu
                
                elif selected_model == "TextBlob":
                    # TextBlob Sentiment Analysis
                    blob = TextBlob(user_input)
                    polarity = blob.sentiment.polarity  # -1 to 1
                    
                    # Convert polarity to sentiment
                    if polarity > 0.1:
                        hf_label = 'POSITIVE'
                        hf_conf = (polarity * 100)
                    elif polarity < -0.1:
                        hf_label = 'NEGATIVE'
                        hf_conf = (polarity * 100)
                    else:
                        hf_label = 'NEUTRAL'
                        hf_conf = ((1 - abs(polarity)) * 100)
                    
                    # Calculate confidence scores
                    pos_score = max(0, polarity * 100)
                    neg_score = max(0, abs(polarity) * 100) if polarity < 0 else 0
                    neu_score = max(0, (1 - abs(polarity)) * 100)
                    
                    total = pos_score + neg_score + neu_score if (pos_score + neg_score + neu_score) > 0 else 1
                    sentiment_scores = {
                        'POSITIVE': (pos_score / total * 100),
                        'NEGATIVE': (neg_score / total * 100),
                        'NEUTRAL': (neu_score / total * 100)
                    }
                
                st.markdown("")
                
                # Main prediction result
                col1, col2 = st.columns([1.5, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%); border: 2px solid rgba(102,126,234,0.3); border-radius: 15px;">
                        <h3 style="color: #a5b4fc; margin-top: 0; font-size: 0.95em; text-transform: uppercase; letter-spacing: 2px;">Predicted Sentiment</h3>
                        <h1 style="font-size: 3em; margin: 15px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{hf_label}</h1>
                        <p style="color: #cbd5e0; font-size: 1.2em; margin: 0;">Confidence: <span style="color: #667eea; font-weight: bold; font-size: 1.3em;">{hf_conf:.1f}%</span></p>
                        <p style="color: #a0aec0; font-size: 0.85em; margin-top: 10px;">{selected_model} Model</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 0 0 0 30px;">
                        <p style="color: #a5b4fc; font-size: 0.95em; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 25px;">Confidence Breakdown</p>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence bars
                    sentiment_order = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
                    colors_map = {'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444', 'NEUTRAL': '#8b5cf6'}
                    emojis_map = {'POSITIVE': '😊', 'NEGATIVE': '😞', 'NEUTRAL': '😐'}
                    
                    for sentiment in sentiment_order:
                        confidence = sentiment_scores.get(sentiment, 0)
                        color = colors_map[sentiment]
                        emoji = emojis_map[sentiment]
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="color: #cbd5e0; font-weight: 600;">{emoji} {sentiment}</span>
                                <span style="color: {color}; font-weight: bold;">{confidence:.1f}%</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 8px; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, {color} 0%, {color}99 100%); height: 100%; width: {confidence}%; transition: width 0.3s ease;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Chart visualization with attractive design
                st.markdown(f"""
                <p style="color: #a5b4fc; font-size: 0.95em; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;">Confidence Distribution</p>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Plotly Bar Chart (medium size, dark theme)
                    sentiments = list(sentiment_scores.keys())
                    scores = list(sentiment_scores.values())
                    colors = [colors_map[s] for s in sentiments]

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=sentiments,
                        y=scores,
                        marker=dict(color=colors, line=dict(color='#667eea', width=1)),
                        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
                    ))

                    # annotations for values
                    for x, y in zip(sentiments, scores):
                        fig_bar.add_annotation(x=x, y=y + 2, text=f"{y:.1f}%", showarrow=False, font=dict(color='#e0e7ff', size=12))

                    fig_bar.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                        font_color='#e0e7ff', height=420, margin=dict(t=30,b=30,l=40,r=20))
                    fig_bar.update_yaxes(range=[0,105], title_text='Confidence (%)', gridcolor='#4a5568')
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col2:
                    # Plotly Donut Chart
                    sentiments = list(sentiment_scores.keys())
                    scores = list(sentiment_scores.values())
                    colors = [colors_map[s] for s in sentiments]

                    fig_pie = go.Figure(go.Pie(labels=sentiments, values=scores, hole=0.45,
                                            marker=dict(colors=colors, line=dict(color='#667eea', width=2)),
                                            textinfo='percent'))
                    fig_pie.update_traces(textfont_size=12)
                    fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                        font_color='#e0e7ff', height=420, title_text='Sentiment Distribution', title_x=0.5,
                                        margin=dict(t=40,b=20,l=20,r=20))
                    st.plotly_chart(fig_pie, use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ Error in sentiment analysis: {str(e)}")
        
        elif not predict_button and not user_input.strip():
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border: 2px solid rgba(102,126,234,0.2); border-radius: 15px; margin: 20px 0;">
                <h3 style="color: #a5b4fc; margin-top: 0;">👆 Start Analyzing</h3>
                <p style="color: #cbd5e0; font-size: 1.1em;">Enter some text above and click the "Predict Sentiment" button to get instant sentiment predictions with confidence scores</p>
            </div>
            """, unsafe_allow_html=True)
    elif predict_button and len(user_input.strip()) < 10:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.error("⚠️ Please enter at least 10 characters to perform sentiment analysis.")
        
# EDA Dashboard Page
elif page == "📈 EDA Dashboard":
    st.markdown("### 📊 Exploratory Data Analysis Dashboard")
    
    # Tab selection
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(
        ["Sentiment Overview", "Rating Analysis", "Location Analysis", "Review Insights"]
    )
    
    # Tab 1: Sentiment Overview
    with eda_tab1:
        st.markdown("#### Overall Sentiment Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            sentiment_counts = data['sentiment'].value_counts()
            colors = ['#10b981', '#ef4444', '#8b5cf6']
            fig_p = go.Figure(go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=0.35,
                                     marker=dict(colors=colors, line=dict(color='#667eea', width=1)),
                                     textinfo='percent+label'))
            fig_p.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                font_color='#e0e7ff', height=420, title_text='Sentiment Distribution', title_x=0.5,
                                margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_p, use_container_width=True)
        
        with col2:
            st.markdown("**Sentiment Statistics:**")
            sentiment_stats = data['sentiment'].value_counts()
            for sentiment, count in sentiment_stats.items():
                percentage = (count / len(data)) * 100
                st.write(f"**{sentiment}**: {count} reviews ({percentage:.1f}%)")
        
        st.markdown("---")
        
        # Sentiment over time
        st.markdown("#### Sentiment Trends Over Time")
        
        data['date_clean'] = pd.to_datetime(data['date'], errors='coerce')
        data_with_dates = data.dropna(subset=['date_clean'])
        
        if len(data_with_dates) > 0:
            data_with_dates['month'] = data_with_dates['date_clean'].dt.to_period('M')
            sentiment_trend = data_with_dates.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            sentiment_trend.index = sentiment_trend.index.to_timestamp()
            
            colors_dict = {'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#8b5cf6'}
            fig_trend = go.Figure()
            for column in sentiment_trend.columns:
                fig_trend.add_trace(go.Scatter(
                    x=sentiment_trend.index.astype(str),
                    y=sentiment_trend[column],
                    mode='lines+markers',
                    name=column,
                    line=dict(color=colors_dict.get(column, '#667eea'), width=3),
                    marker=dict(size=6)
                ))

            fig_trend.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                    font_color='#e0e7ff', height=420, title_text='Monthly Sentiment Trends', title_x=0.5,
                                    xaxis=dict(title='Month', tickangle=45), yaxis=dict(title='Number of Reviews', gridcolor='#4a5568'))
            st.plotly_chart(fig_trend, use_container_width=True)
    
    # Tab 2: Rating Analysis
    with eda_tab2:
        st.markdown("#### Sentiment Distribution by Rating")
        
        rating_sentiment = pd.crosstab(data['rating'], data['sentiment'], normalize='index') * 100
        
        # Plotly grouped bar chart for rating vs sentiment
        fig_rs = go.Figure()
        palette = ['#ef4444', '#8b5cf6', '#10b981']
        for i, col in enumerate(rating_sentiment.columns):
            fig_rs.add_trace(go.Bar(x=rating_sentiment.index.astype(str), y=rating_sentiment[col], name=col,
                                    marker_color=palette[i % len(palette)]))

        fig_rs.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                             font_color='#e0e7ff', height=420, title_text='Sentiment Distribution by Star Rating (%)', title_x=0.5,
                             xaxis=dict(title='Star Rating'), yaxis=dict(title='Percentage of Reviews', gridcolor='#4a5568'))
        fig_rs.update_xaxes(tickangle=0)
        st.plotly_chart(fig_rs, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### Review Length Distribution by Sentiment")
        
        # Plotly box plot for Review Length by Sentiment
        color_map = {'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#8b5cf6'}
        fig_box = px.box(data, x='sentiment', y='review_length', color='sentiment',
                         color_discrete_map=color_map, points='outliers')
        fig_box.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                              font_color='#e0e7ff', height=420, title_text='Review Length Distribution by Sentiment', title_x=0.5,
                              yaxis=dict(gridcolor='#4a5568'))
        fig_box.update_xaxes(title_text='Sentiment')
        fig_box.update_yaxes(title_text='Review Length')
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Tab 3: Location Analysis
    with eda_tab3:
        st.markdown("#### Top Locations by Sentiment")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Top 10 Locations with Most Positive Reviews:**")
            
            location_sentiment = data.groupby(['location', 'sentiment']).size().unstack(fill_value=0)
            top_positive = location_sentiment.sort_values(by='Positive', ascending=False).head(10)
            
            # Plotly horizontal bar for top positive locations
            fig_pos = px.bar(x=top_positive['Positive'], y=top_positive.index, orientation='h',
                             color_discrete_sequence=['#10b981'])
            fig_pos.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                  font_color='#e0e7ff', height=420, title_text='Top 10 Locations - Positive Reviews', title_x=0.5,
                                  xaxis=dict(title='Number of Reviews', gridcolor='#4a5568'))
            fig_pos.update_yaxes(autorange='reversed')
            st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Locations with Most Negative Reviews:**")
            
            top_negative = location_sentiment.sort_values(by='Negative', ascending=False).head(10)
            
            # Plotly horizontal bar for top negative locations
            fig_neg = px.bar(x=top_negative['Negative'], y=top_negative.index, orientation='h',
                             color_discrete_sequence=['#ef4444'])
            fig_neg.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                  font_color='#e0e7ff', height=420, title_text='Top 10 Locations - Negative Reviews', title_x=0.5,
                                  xaxis=dict(title='Number of Reviews', gridcolor='#4a5568'))
            fig_neg.update_yaxes(autorange='reversed')
            st.plotly_chart(fig_neg, use_container_width=True)
    
    # Tab 4: Review Insights
    with eda_tab4:
        st.markdown("#### Verified Purchase Sentiment Analysis")
        
        verified_sentiment = data.groupby(['verified_purchase', 'sentiment']).size().unstack(fill_value=0)
        verified_sentiment_pct = verified_sentiment.div(verified_sentiment.sum(axis=1), axis=0) * 100
        
        # Plotly stacked bar for verified vs non-verified sentiment percentages
        fig_vs = go.Figure()
        palette = ['#ef4444', '#8b5cf6', '#10b981']
        for i, col in enumerate(verified_sentiment_pct.columns):
            fig_vs.add_trace(go.Bar(x=verified_sentiment_pct.index.astype(str), y=verified_sentiment_pct[col], name=col,
                                     marker_color=palette[i % len(palette)]))

        fig_vs.update_layout( template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                             font_color='#e0e7ff', height=420, title_text='Sentiment Distribution: Verified vs Non-Verified Purchases', title_x=0.5,
                             xaxis=dict(title='Verified Purchase'), yaxis=dict(title='Percentage of Reviews (%)', gridcolor='#4a5568'))
        fig_vs.update_xaxes(tickangle=0)
        st.plotly_chart(fig_vs, use_container_width=True)

# Advanced Analytics Page
elif page == "📉 Advanced Analytics":
    st.markdown("### 📊 Advanced Analytics & Insights")
    
    analytics_tab1, analytics_tab2 = st.tabs(["Platform Analysis", "Keyword Insights"])
    
    with analytics_tab1:
        st.markdown("#### Platform-wise Sentiment Analysis")
        
        # Check if platform_type column exists, if not create it
        if 'platform_type' not in data.columns:
            # Create a simplified platform_type based on platform column
            platform_mapping = {'Flipkart': 'Mobile', 'Amazon': 'Web', 'App Store': 'Mobile', 'Website': 'Web'}
            data['platform_type'] = data['platform'].map(platform_mapping).fillna('Other')
        
        platform_sentiment = data.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
        
        # Plotly grouped bar for platform sentiment
        fig_plat = go.Figure()
        palette = ['#ef4444', '#8b5cf6', '#10b981']
        for i, col in enumerate(platform_sentiment.columns):
            fig_plat.add_trace(go.Bar(x=platform_sentiment.index.astype(str), y=platform_sentiment[col], name=col,
                                      marker_color=palette[i % len(palette)]))

        fig_plat.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                               font_color='#e0e7ff', height=420, title_text='Sentiment Distribution Across Platforms', title_x=0.5,
                               xaxis=dict(title='Platform', tickangle=45), yaxis=dict(title='Number of Reviews', gridcolor='#4a5568'))
        st.plotly_chart(fig_plat, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### Version-wise Sentiment Performance")
        
        version_sentiment = data.groupby(['version', 'sentiment']).size().unstack(fill_value=0)
        version_sentiment['total'] = version_sentiment.sum(axis=1)
        version_sentiment['pos_pct'] = (version_sentiment.get('Positive', 0) / version_sentiment['total']) * 100
        
        top_versions = version_sentiment.sort_values(by='total', ascending=False).head(12)
        
        # Plotly bar with color scale for positive sentiment % per version
        df_versions = top_versions.reset_index()
        # reset_index() puts the version label in the 'version' column — use that as x
        fig_versions = px.bar(df_versions, x='version', y='pos_pct', color='pos_pct', color_continuous_scale='RdYlGn',
                  range_color=(0,100), labels={'version': 'Version', 'pos_pct': 'Positive Sentiment (%)'})
        fig_versions.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                       font_color='#e0e7ff', height=420, title_text='Positive Sentiment % for Top 12 Reviewed Versions', title_x=0.5,
                       xaxis=dict(tickangle=45))
        fig_versions.update_coloraxes(showscale=False)
        st.plotly_chart(fig_versions, use_container_width=True)
    
    with analytics_tab2:
        st.markdown("#### Top Keywords in Negative Reviews")
        
        negative_reviews = data[data['sentiment'] == 'Negative']['processed_reviews']
        
        if len(negative_reviews) > 0:
            vec = CountVectorizer(stop_words='english', max_features=15)
            matrix = vec.fit_transform(negative_reviews)
            counts = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out()).sum().sort_values(ascending=True)
            
            # Plotly horizontal bar for top keywords in negative reviews
            fig_kw = px.bar(x=counts.values, y=counts.index, orientation='h',
                            color_discrete_sequence=['#ef4444'])
            fig_kw.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                 font_color='#e0e7ff', height=420, title_text='Top Keywords in Negative Reviews', title_x=0.5,
                                 xaxis=dict(title='Frequency', gridcolor='#4a5568'))
            fig_kw.update_yaxes(automargin=True)
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("No negative reviews found in dataset.")
        
        st.markdown("---")
        
        st.markdown("#### Rating Distribution")
        
        # Plotly bar for rating distribution
        rating_counts = data['rating'].value_counts().sort_index()
        fig_rating = go.Figure()
        fig_rating.add_trace(go.Bar(x=rating_counts.index.astype(str), y=rating_counts.values,
                                    marker_color='#4f8bd4', text=rating_counts.values, textposition='outside'))
        fig_rating.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a202c',
                                 font_color='#e0e7ff', height=420, title_text='Distribution of Star Ratings', title_x=0.5,
                                 xaxis=dict(title='Rating'), yaxis=dict(title='Number of Reviews', gridcolor='#4a5568'))
        st.plotly_chart(fig_rating, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; font-size: 0.9em; padding: 20px;">
    <p style="margin: 5px 0;">🎯 AI Echo - Sentiment Analysis Platform | Built with Streamlit & Python</p>
    <p style="margin: 5px 0;">Using BERT, VADER, and TextBlob for multi-model sentiment analysis</p>
    <p style="margin: 5px 0; color: #718096; font-size: 0.8em;">Dark Mode Theme | Enhanced Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)
