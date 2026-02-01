# Text Classification System - SpamDetection

## Project Overview
A comprehensive machine learning system for text classification built as part of the Ardentix AI/ML Engineer Intern selection process. This system demonstrates a complete ML pipeline from data preprocessing to model deployment with multiple interfaces and deployment options.

## 🚀 Features
- **Complete ML Pipeline**: Text preprocessing, feature extraction, model training, evaluation
- **Multiple Algorithms**: Naive Bayes, Logistic Regression, Random Forest, SVM
- **Feature Extraction**: TF-IDF and Bag of Words vectorization
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC curves
- **Interactive Web Interface**: Streamlit application with real-time predictions
- **REST API**: FastAPI backend for integrations
- **Multiple Datasets**: SMS Spam, 20 Newsgroups, Movie Reviews
- **Deployment Ready**: Docker, Heroku, Streamlit Cloud configurations
- **Visualization**: Word clouds, confusion matrices, feature importance

## 📁 Project Structure
```
SpamDetection/
├── src/                    # Core source code
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── models.py          # ML model implementations
│   ├── evaluation.py      # Model evaluation metrics
│   └── visualization.py   # Data visualization tools
├── data/                  # Dataset management
│   └── dataset_loader.py  # Dataset loading utilities
├── notebooks/             # Jupyter notebooks for exploration
├── .github/workflows/     # CI/CD pipeline
├── app.py                # Streamlit web interface
├── deploy_api.py         # FastAPI REST API
├── main.py               # Main execution script
├── demo.py               # Quick demonstration
├── deploy.py             # Deployment helper script
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── DEPLOYMENT.md         # Detailed deployment guide
```

## 🎯 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/nithin-kata/SpamDetection.git
cd SpamDetection

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

## 🚀 Deployment

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
1. Push to GitHub ✅ (Done!)
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

## 📊 Datasets
- **SMS Spam Collection**: Binary classification (ham/spam)
- **20 Newsgroups**: Multi-class text categorization  
- **Movie Reviews**: Sentiment analysis (positive/negative)

## 🎓 Educational Value
This project demonstrates:
- **Machine Learning Fundamentals**: Complete supervised learning pipeline
- **Text Processing**: NLP preprocessing and feature engineering
- **Model Evaluation**: Comprehensive metrics and comparison
- **Software Engineering**: Modular design, testing, documentation
- **Deployment**: Multiple deployment strategies and platforms
- **Web Development**: Interactive applications and REST APIs

## 📈 Performance Results
- **Logistic Regression**: ~90% accuracy on movie reviews
- **Naive Bayes**: ~65% accuracy (baseline)
- **Real-time Prediction**: Sub-second response times
- **Scalable Architecture**: Supports batch processing

## 🛠️ Technical Stack
- **Python 3.10+**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Streamlit**: Web application framework
- **FastAPI**: REST API framework
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline

## 📝 Documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Comprehensive deployment guide
- [DOCUMENTATION.md](DOCUMENTATION.md) - Technical documentation
- [Jupyter Notebook](notebooks/text_classification_exploration.ipynb) - Interactive exploration

## 🤝 Contributing
This project was built for the Ardentix AI/ML Engineer Intern selection process, showcasing production-ready machine learning engineering skills.

---
**Built with ❤️ for Ardentix Internship Application**
