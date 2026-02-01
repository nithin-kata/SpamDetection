# Text Classification System - Complete Documentation

## Overview

This text classification system is designed for the Ardentix AI/ML Engineer Intern assignment. It demonstrates a complete machine learning pipeline from data preprocessing to model deployment, showcasing proficiency in Python, machine learning fundamentals, and software engineering best practices.

## Project Architecture

```
text-classification/
├── src/                        # Core source code
│   ├── preprocessing.py        # Text preprocessing utilities
│   ├── models.py              # ML model implementations
│   ├── evaluation.py          # Model evaluation metrics
│   └── visualization.py       # Data visualization tools
├── data/                      # Dataset management
│   └── dataset_loader.py      # Dataset loading utilities
├── notebooks/                 # Jupyter notebooks
│   └── text_classification_exploration.ipynb
├── models/                    # Saved model artifacts (created at runtime)
├── main.py                    # Main execution script
├── app.py                     # Streamlit web application
├── test_system.py            # System verification tests
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## Core Components

### 1. Text Preprocessing (`src/preprocessing.py`)

**TextPreprocessor Class:**
- Text cleaning (URLs, emails, special characters)
- Tokenization using NLTK
- Stopword removal
- Stemming with Porter Stemmer
- Complete preprocessing pipeline

**FeatureExtractor Class:**
- TF-IDF vectorization with configurable parameters
- Bag of Words feature extraction
- N-gram support (unigrams and bigrams)
- Feature transformation for new texts

### 2. Machine Learning Models (`src/models.py`)

**TextClassifier Class:**
- Unified interface for multiple algorithms:
  - Naive Bayes (MultinomialNB)
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Model training and prediction
- Feature importance extraction
- Model persistence (save/load)

**ModelTuner Class:**
- Hyperparameter optimization using GridSearchCV
- Predefined parameter grids for each model type
- Cross-validation support
- Best parameter selection

### 3. Model Evaluation (`src/evaluation.py`)

**ModelEvaluator Class:**
- Comprehensive metrics calculation:
  - Accuracy, Precision, Recall, F1-Score
  - Macro and weighted averages
  - AUC-ROC for binary classification
- Classification reports
- Confusion matrix visualization
- ROC curve plotting
- Model comparison utilities
- Best model selection

### 4. Visualization (`src/visualization.py`)

**TextVisualization Class:**
- Class distribution plots (bar charts, pie charts)
- Text length analysis by class
- Word clouds for each class
- Feature importance visualization
- Learning curves
- Interactive confusion matrices (Plotly)
- Prediction confidence analysis
- Calibration plots

### 5. Dataset Management (`data/dataset_loader.py`)

**Available Datasets:**
- **SMS Spam Detection**: Binary classification (ham/spam)
- **20 Newsgroups**: Multi-class text categorization
- **Movie Reviews**: Sentiment analysis (positive/negative)

**Features:**
- Automatic dataset downloading
- Consistent data format across datasets
- Easy dataset switching
- Label mapping and metadata

## Usage Guide

### 1. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run system tests
python test_system.py

# Execute complete pipeline
python main.py

# Launch web interface
streamlit run app.py
```

### 2. Command Line Usage

```python
from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import TextClassifier
from src.evaluation import ModelEvaluator

# Initialize components
preprocessor = TextPreprocessor()
feature_extractor = FeatureExtractor()
evaluator = ModelEvaluator()

# Load and preprocess data
texts, labels, label_names = load_dataset('sms_spam')
processed_texts = [preprocessor.preprocess_text(text) for text in texts]

# Extract features
X_train_tfidf = feature_extractor.extract_tfidf_features(processed_texts)

# Train model
classifier = TextClassifier('naive_bayes')
classifier.train(X_train_tfidf, labels)

# Evaluate
metrics = evaluator.evaluate_model(classifier, X_test_tfidf, y_test, "NB_Model")
```

### 3. Web Interface Features

The Streamlit application provides:
- **Home Page**: Project overview and statistics
- **Data Exploration**: Interactive data analysis and visualization
- **Model Training**: Configure and train models with real-time feedback
- **Prediction**: Real-time text classification with confidence scores
- **Results Analysis**: Comprehensive model comparison and evaluation

### 4. Jupyter Notebook

The exploration notebook (`notebooks/text_classification_exploration.ipynb`) provides:
- Step-by-step pipeline walkthrough
- Interactive data exploration
- Model comparison and analysis
- Visualization examples
- Educational content with explanations

## Technical Implementation Details

### Text Preprocessing Pipeline

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs, emails, phone numbers
   - Remove special characters and digits
   - Normalize whitespace

2. **Tokenization**:
   - NLTK word tokenization
   - Handle punctuation and contractions

3. **Stopword Removal**:
   - English stopwords from NLTK
   - Customizable stopword lists

4. **Stemming**:
   - Porter Stemmer for word normalization
   - Optional stemming toggle

### Feature Extraction

**TF-IDF Vectorization**:
- Term Frequency-Inverse Document Frequency
- N-gram support (1-2 grams)
- Min/max document frequency filtering
- Configurable vocabulary size

**Bag of Words**:
- Simple word count vectorization
- Same filtering and n-gram options
- Comparison baseline for TF-IDF

### Model Selection Rationale

**Naive Bayes**:
- Excellent baseline for text classification
- Fast training and prediction
- Works well with limited data
- Probabilistic output interpretation

**Logistic Regression**:
- Linear model with good interpretability
- Regularization options (L1/L2)
- Robust performance across datasets
- Feature importance via coefficients

**Additional Models** (available but not in main pipeline):
- Random Forest: Ensemble method, handles non-linear patterns
- SVM: Effective for high-dimensional text data

### Evaluation Methodology

**Metrics Used**:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Binary classification performance

**Evaluation Strategy**:
- Train/test split with stratification
- Multiple model comparison
- Feature extraction method comparison
- Best model selection based on F1-score

## Performance Characteristics

### Expected Results

**SMS Spam Dataset**:
- Naive Bayes: ~95% accuracy
- Logistic Regression: ~96% accuracy
- TF-IDF generally outperforms BoW

**20 Newsgroups Dataset**:
- Naive Bayes: ~85% accuracy
- Logistic Regression: ~88% accuracy
- More challenging due to topic similarity

**Movie Reviews Dataset**:
- Both models: ~85-90% accuracy
- Sentiment analysis baseline performance

### Computational Complexity

- **Preprocessing**: O(n*m) where n=documents, m=avg length
- **Feature Extraction**: O(n*v) where v=vocabulary size
- **Training**: Varies by algorithm
  - Naive Bayes: O(n*v)
  - Logistic Regression: O(n*v*i) where i=iterations
- **Prediction**: O(v) per document

## Extensibility and Future Enhancements

### Immediate Improvements

1. **Advanced Preprocessing**:
   - Lemmatization instead of stemming
   - Named entity recognition
   - Custom tokenization rules

2. **Feature Engineering**:
   - Word embeddings (Word2Vec, GloVe)
   - Character-level n-grams
   - Syntactic features (POS tags)

3. **Model Enhancements**:
   - Deep learning models (LSTM, BERT)
   - Ensemble methods
   - Active learning for data efficiency

4. **Evaluation Improvements**:
   - Cross-validation
   - Statistical significance testing
   - Error analysis tools

### Production Considerations

1. **Scalability**:
   - Batch processing for large datasets
   - Incremental learning capabilities
   - Distributed computing support

2. **Deployment**:
   - REST API development
   - Model versioning
   - A/B testing framework

3. **Monitoring**:
   - Performance tracking
   - Data drift detection
   - Model retraining triggers

## Dependencies and Requirements

### Core Dependencies
- **Python**: 3.8+
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing

### Visualization
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive plots
- **WordCloud**: Text visualization

### Web Interface
- **Streamlit**: Web application framework

### Development
- **Jupyter**: Interactive development
- **pytest**: Testing framework (optional)

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

2. **Memory Issues with Large Datasets**:
   - Reduce `max_features` parameter
   - Use batch processing
   - Consider sparse matrix operations

3. **Poor Model Performance**:
   - Check data quality and preprocessing
   - Verify class balance
   - Try different feature extraction methods
   - Tune hyperparameters

4. **Streamlit Issues**:
   - Clear cache: `streamlit cache clear`
   - Restart application
   - Check port availability

### Performance Optimization

1. **Speed Improvements**:
   - Use sparse matrices for features
   - Implement parallel processing
   - Cache preprocessed data

2. **Memory Optimization**:
   - Process data in chunks
   - Use generators for large datasets
   - Optimize feature matrix storage

## Conclusion

This text classification system demonstrates a comprehensive understanding of the machine learning pipeline, from data preprocessing to model evaluation. The modular design allows for easy extension and modification, while the multiple interfaces (command line, web app, notebook) cater to different use cases and user preferences.

The implementation showcases best practices in:
- Code organization and modularity
- Documentation and testing
- User interface design
- Machine learning methodology
- Software engineering principles

This system serves as a solid foundation for more advanced text classification projects and demonstrates readiness for real-world machine learning challenges.