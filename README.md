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
<img width="1918" height="867" alt="Screenshot 2026-02-01 150537" src="https://github.com/user-attachments/assets/8d412457-6a89-4ccb-ac7d-7bfb4981b1be" />
<img width="1919" height="868" alt="Screenshot 2026-02-01 150550" src="https://github.com/user-attachments/assets/85079d35-779e-4f4d-9863-c5d2a1c62ed3" />
<img width="1497" height="863" alt="Screenshot 2026-02-01 151035" src="https://github.com/user-attachments/assets/c2192964-5a70-4531-954d-d82824ad33d7" />
<img width="1440" height="855" alt="Screenshot 2026-02-01 151115" src="https://github.com/user-attachments/assets/4a5c5647-3ff5-456d-b111-aca4b4bd3236" />
<img width="1478" height="851" alt="Screenshot 2026-02-01 151236" src="https://github.com/user-attachments/assets/3986599e-38d9-49ff-a0c6-673bb271892f" />
<img width="1477" height="862" alt="Screenshot 2026-02-01 151350" src="https://github.com/user-attachments/assets/2d475f8f-48f8-415f-8c2c-3110dc567d91" />
<img width="1442" height="829" alt="Screenshot 2026-02-01 151410" src="https://github.com/user-attachments/assets/eb2d3070-6fae-405a-8641-085f1cb8a532" />
<img width="1484" height="809" alt="Screenshot 2026-02-01 151433" src="https://github.com/user-attachments/assets/3ce0302e-31b4-44b8-baff-210422b7483a" />
<img width="1600" height="843" alt="Screenshot 2026-02-01 151544" src="https://github.com/user-attachments/assets/dbbc2dd2-fdf0-489e-aa7f-f2402ffee65a" />
<img width="1432" height="624" alt="Screenshot 2026-02-01 151552" src="https://github.com/user-attachments/assets/c1ac80cf-ce36-4cbd-86f4-7ccac512ded6" />




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

