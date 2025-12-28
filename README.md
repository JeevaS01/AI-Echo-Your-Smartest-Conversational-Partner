# 🎯 AI Echo - Your Smartest Conversational Partner

A powerful, multi-model sentiment analysis platform built with Streamlit, featuring real-time predictions, exploratory data analysis, and advanced analytics using VADER, TextBlob, and BERT models.

---

## ✨ Features

### 🏠 **Home Dashboard**
- Quick sentiment overview with distribution charts
- Dataset statistics (total reviews, locations, platforms, average rating)
- Sentiment breakdown with visual cards

![App Screenshot](image/home.png)

### 🔍 **Real-time Sentiment Analysis**
- Multi-model sentiment prediction (VADER, TextBlob, BERT)
- Confidence scores and breakdown visualization
- Interactive bar and donut charts
- Real-time text input analysis

![Real-time Analysis](image/p2.png)
![Real-time Analysis](image/p2.1.png)
![Real-time Analysis](image/p2.2.png)
![Real-time Analysis](image/p2.3.png)
![Real-time Analysis](image/p2.4.png)

### 📊 **EDA Dashboard**
- **Sentiment Overview**: Distribution trends and monthly sentiment trends
- **Rating Analysis**: Sentiment by star rating, review length distribution
- **Location Analysis**: Top locations by positive/negative sentiment
- **Review Insights**: Verified purchase sentiment comparison

![EDA Dashboard](docs/images/eda-dashboard.png)

### 📈 **Advanced Analytics**
- Platform-wise sentiment distribution
- Version-wise sentiment performance
- Keyword analysis from negative reviews
- Rating distribution visualization

![Advanced Analytics](docs/images/advanced-analytics.png)

---

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd "GUVI DS/NLP/ai echo"
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Place your data file**
- Ensure `cleaned_review.csv` is in the project root or `data/` folder
- Required columns: `location`, `platform`, `rating`, `sentiment`, `date`, `review_length`, `processed_reviews`, `version`, `verified_purchase`

5. **Run the app**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8502`

---

## 📦 Project Structure

```
ai echo/
├── streamlit_app.py          # Main application
├── cleaned_review.csv        # Dataset (place in root or data/ folder)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit Cloud configuration
├── style.css                # Custom styling
├── eda.py                   # EDA utilities
└── docs/
    └── images/              # Screenshot placeholders
        ├── banner.png
        ├── home-dashboard.png
        ├── real-time-analysis.png
        ├── eda-dashboard.png
        └── advanced-analytics.png
```

---

## 📋 Data Requirements

Your `cleaned_review.csv` should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `location` | string | Review location |
| `platform` | string | Platform (Amazon, Flipkart, etc.) |
| `rating` | int | Star rating (1-5) |
| `sentiment` | string | Sentiment class (Positive, Negative, Neutral) |
| `date` | datetime | Review date |
| `review_length` | int | Length of review |
| `processed_reviews` | string | Cleaned/processed review text |
| `version` | string | Product version |
| `verified_purchase` | bool | Verified purchase flag |

---

## 🤖 Sentiment Analysis Models

### VADER (Recommended)
- ⚡ Fast, rule-based sentiment analyzer
- 📊 Works well for social media & reviews
- ✅ No heavy dependencies

### TextBlob
- 🎯 Lightweight polarity-based analysis
- 📚 Uses pre-trained models
- 🔄 Graceful fallback when others fail

### BERT (Optional)
- 🧠 Deep learning-based transformer model
- 🎓 Most accurate but requires `transformers` package
- ⚠️ Falls back to TextBlob on Cloud if not installed

---

## ☁️ Deployment on Streamlit Cloud

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Deploy AI Echo to Streamlit Cloud"
git push origin main
```

### Step 2: Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repo, branch, and `streamlit_app.py`
5. Click "Deploy"

### Configuration
The `.streamlit/config.toml` file is pre-configured for Cloud:
- ✅ Dark theme matching your UI
- ✅ Optimized server settings
- ✅ CORS enabled
- ✅ Headless mode activated

### Notes for Cloud
- **Data file**: Ensure `cleaned_review.csv` is committed to git
- **BERT**: Not included by default (heavy dependency). The app will use TextBlob fallback instead
- **Build time**: First deploy takes ~2-3 minutes

---

## 🎨 Theme & Customization

The app uses a **dark gradient theme** with:
- Primary color: `#667eea` (purple)
- Background: `#0f1419` (dark blue)
- Accent colors for sentiment: 
  - 🟢 Positive: `#10b981`
  - 🔴 Negative: `#ef4444`
  - 🟣 Neutral: `#8b5cf6`

Edit the CSS in `streamlit_app.py` to customize colors and styling.

---

## 📊 Sample Dashboard Walkthrough

### Home Tab
![Home Tab](docs/images/home-tab.png)
- View overall sentiment distribution
- See dataset metrics at a glance
- Identify positive/negative/neutral review counts

### Real-time Analysis Tab
![Real-time Tab](docs/images/real-time-tab.png)
1. Select a sentiment model (VADER recommended)
2. Type or paste text into the input area
3. Click "Predict Sentiment"
4. View confidence breakdown and charts

### EDA Dashboard Tab
![EDA Tab](docs/images/eda-tab.png)
- Explore sentiment trends over time
- Analyze sentiment by star rating
- Compare locations and platforms

### Advanced Analytics Tab
![Analytics Tab](docs/images/analytics-tab.png)
- Platform performance comparison
- Keyword analysis from negative reviews
- Version-wise sentiment tracking

---

## 🛠️ Troubleshooting

### Data not loading?
- Check if `cleaned_review.csv` is in the project root or `data/` folder
- Verify CSV has required columns (see Data Requirements)
- The app will show a fallback single-row dataset if file is missing

### BERT model not loading?
- This is expected in Cloud environments (heavy dependency)
- The app automatically falls back to TextBlob
- To enable BERT locally, install: `pip install transformers torch`

### App running slowly?
- Clear Streamlit cache: Delete `.streamlit/` in user directory
- Reduce dataset size or filter to recent reviews
- Restart the app: Stop and run `streamlit run streamlit_app.py` again

---

## 📚 Dependencies

See `requirements.txt` for all packages:
- **streamlit**: Web framework
- **pandas, numpy**: Data processing
- **plotly**: Interactive visualizations
- **scikit-learn**: ML utilities
- **nltk, textblob, vaderSentiment**: Sentiment analysis
- **matplotlib, seaborn**: Static visualizations

Optional (for BERT support):
```bash
pip install transformers torch
```

---

## 🔗 Links

- 🌐 **Live Demo**: [AI Echo on Streamlit Cloud](https://your-cloud-url)
- 📖 **Documentation**: See inline code comments
- 🐛 **Issues**: Create an issue in the GitHub repo

---

## 📝 License

This project is open source. Feel free to use and modify!

---

## 👨‍💻 Author

Built with ❤️ for sentiment analysis enthusiasts.

**Version**: 1.0.0  
**Last Updated**: December 28, 2025

---

## 🎓 Dataset Attribution

If using public datasets like Amazon/Flipkart reviews, ensure proper attribution per the dataset's license.

---

## 🚀 Future Enhancements

- [ ] Add support for multi-language sentiment analysis
- [ ] Implement aspect-based sentiment analysis
- [ ] Add export functionality (CSV, PDF reports)
- [ ] Real-time data ingestion from APIs
- [ ] User feedback loop for model improvement
- [ ] Custom model fine-tuning interface

---

**Happy Analyzing! 🎯**
