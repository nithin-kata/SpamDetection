# Deployment Guide - Text Classification System

## Overview
This guide covers multiple deployment options for your text classification system, from local development to cloud production deployment.

## Deployment Options

### 1. Local Development Deployment
**Best for**: Testing, development, presentations

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### 2. Streamlit Cloud Deployment (Recommended)
**Best for**: Quick, free cloud deployment with minimal setup

#### Steps:
1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/text-classification.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration** (if needed):
   Create `.streamlit/config.toml`:
   ```toml
   [server]
   headless = true
   port = 8501
   
   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   ```

### 3. Heroku Deployment
**Best for**: More control, custom domains, scaling options

#### Files needed:

**Procfile**:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
enableCORS=false
port = \$PORT
" > ~/.streamlit/config.toml
```

**runtime.txt**:
```
python-3.10.12
```

#### Deployment Steps:
1. **Install Heroku CLI**
2. **Create Heroku app**:
   ```bash
   heroku create your-text-classifier
   ```
3. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### 4. Docker Deployment
**Best for**: Containerized deployment, Kubernetes, cloud platforms

#### Build and run locally:
```bash
# Build the image
docker build -t text-classifier .

# Run the container
docker run -p 8501:8501 text-classifier
```

#### Using Docker Compose:
```bash
docker-compose up --build
```

#### Deploy to cloud platforms:
- **Google Cloud Run**
- **AWS ECS**
- **Azure Container Instances**

### 5. FastAPI REST API Deployment
**Best for**: API integrations, microservices architecture

#### Run locally:
```bash
# Install additional dependencies
pip install fastapi uvicorn

# Run the API
python deploy_api.py
```

API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### API Endpoints:
- `GET /` - Health check
- `POST /predict` - Single text prediction
- `POST /batch_predict` - Multiple text predictions
- `GET /models` - Model information

#### Example API usage:
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was fantastic!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch_predict",
    json=["Great movie!", "Terrible film!"]
)
print(response.json())
```

### 6. Cloud Platform Deployments

#### Google Cloud Platform
1. **Cloud Run** (Recommended):
   ```bash
   # Build and push to Container Registry
   gcloud builds submit --tag gcr.io/PROJECT_ID/text-classifier
   
   # Deploy to Cloud Run
   gcloud run deploy --image gcr.io/PROJECT_ID/text-classifier --platform managed
   ```

2. **App Engine**:
   Create `app.yaml`:
   ```yaml
   runtime: python310
   
   entrypoint: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   
   automatic_scaling:
     min_instances: 0
     max_instances: 10
   ```

#### AWS Deployment
1. **Elastic Beanstalk**:
   ```bash
   # Install EB CLI
   pip install awsebcli
   
   # Initialize and deploy
   eb init
   eb create text-classifier-env
   eb deploy
   ```

2. **ECS with Fargate**:
   - Push Docker image to ECR
   - Create ECS task definition
   - Deploy to Fargate cluster

#### Azure Deployment
1. **Container Instances**:
   ```bash
   # Create resource group
   az group create --name text-classifier-rg --location eastus
   
   # Deploy container
   az container create \
     --resource-group text-classifier-rg \
     --name text-classifier \
     --image your-registry/text-classifier \
     --ports 8501
   ```

### 7. Production Considerations

#### Performance Optimization
- **Model Caching**: Cache loaded models in memory
- **Feature Caching**: Cache preprocessed features
- **Batch Processing**: Process multiple texts together
- **Load Balancing**: Use multiple instances

#### Security
- **HTTPS**: Always use SSL/TLS in production
- **API Keys**: Implement authentication for API access
- **Rate Limiting**: Prevent abuse with rate limits
- **Input Validation**: Sanitize all inputs

#### Monitoring
- **Health Checks**: Implement proper health endpoints
- **Logging**: Log predictions and errors
- **Metrics**: Track response times and accuracy
- **Alerts**: Set up monitoring alerts

#### Scaling
- **Horizontal Scaling**: Multiple instances
- **Auto-scaling**: Based on CPU/memory usage
- **Database**: Store models and results in database
- **CDN**: Cache static assets

### 8. Environment Variables

Create `.env` file for configuration:
```env
# Model settings
MODEL_PATH=models/
DEFAULT_MODEL=logistic_regression_TF-IDF

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/textclassifier

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn
```

### 9. CI/CD Pipeline

#### GitHub Actions example (`.github/workflows/deploy.yml`):
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python test_system.py
    
    - name: Deploy to Streamlit Cloud
      # Add deployment steps here
```

## Quick Start Commands

### Local Development:
```bash
streamlit run app.py
```

### Docker:
```bash
docker-compose up --build
```

### API:
```bash
python deploy_api.py
```

### Heroku:
```bash
git push heroku main
```

Choose the deployment method that best fits your needs:
- **Streamlit Cloud**: Easiest, free tier available
- **Heroku**: More control, custom domains
- **Docker**: Maximum flexibility, works anywhere
- **FastAPI**: For API-based integrations
- **Cloud Platforms**: Enterprise-grade scaling