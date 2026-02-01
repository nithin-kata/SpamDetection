# Text Classification System

## Project Overview
A machine learning system for text classification built as part of the Ardentix AI/ML Engineer Intern selection process.

## Features
- Text preprocessing pipeline (cleaning, tokenization, stopword removal)
- Multiple feature extraction methods (TF-IDF, Bag of Words)
- Classification models (Naive Bayes, Logistic Regression)
- Comprehensive evaluation metrics
- Interactive Streamlit interface
- Visualization of results

## Project Structure
```
text-classification/
├── data/                   # Dataset files
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── models.py          # ML models
│   ├── evaluation.py      # Model evaluation
│   └── visualization.py   # Result visualization
├── notebooks/             # Jupyter notebooks for exploration
├── app.py                # Streamlit web interface
├── requirements.txt      # Dependencies
└── main.py              # Main execution script
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd text-classification

# Install dependencies
pip install -r requirements.txt

# Run system tests
python test_system.py
```

### 2. Usage Options

**Quick Demo** (Recommended first):
```bash
python demo.py
```

**Full Pipeline**:
```bash
python main.py
```

**Web Interface**:
```bash
streamlit run app.py
```

**REST API**:
```bash
python deploy_api.py
```

## Deployment

### Easy Deployment Script
```bash
# Local deployment
python deploy.py local

# Docker deployment
python deploy.py docker

# API deployment
python deploy.py api

# Heroku setup
python deploy.py heroku

# Streamlit Cloud setup
python deploy.py streamlit-cloud
```

### Manual Deployment Options

**Streamlit Cloud** (Recommended):
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

**Docker**:
```bash
docker build -t text-classifier .
docker run -p 8501:8501 text-classifier
```

**Heroku**:
```bash
heroku create your-app-name
git push heroku main
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Datasets
- **SMS Spam Collection**: Binary classification (ham/spam)
- **20 Newsgroups**: Multi-class text categorization  
- **Movie Reviews**: Sentiment analysis (positive/negative)