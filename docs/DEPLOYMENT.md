# FloodLLM Deployment Guide

This guide covers deploying FloodLLM to Google Cloud Platform (GCP) and other production environments.

---

## Prerequisites

- Google Cloud Platform account
- `gcloud` CLI installed
- Docker installed (for containerization)
- Domain name (optional, for custom URLs)

---

## 1. GCP Project Setup

### 1.1 Create a GCP Project

```bash
gcloud projects create flood-llm-PROJECT_ID --name="FloodLLM"
gcloud config set project flood-llm-PROJECT_ID
```

### 1.2 Enable Required APIs

```bash
# Earth Engine API (for satellite data)
gcloud services enable earthengine.googleapis.com

# Cloud Run API (for hosting)
gcloud services enable run.googleapis.com

# Artifact Registry (for container images)
gcloud services enable artifactregistry.googleapis.com

# Cloud Build (for CI/CD)
gcloud services enable cloudbuild.googleapis.com

# Vertex AI API (for LLM hosting if using custom models)
gcloud services enable aiplatform.googleapis.com
```

### 1.3 Set Up Authentication

```bash
gcloud auth login
gcloud auth configure-docker LOCATION-docker.pkg.dev
```

---

## 2. Earth Engine Configuration

### 2.1 Initialize Earth Engine

```bash
earthengine authenticate
```

### 2.2 Create Service Account (for production)

```bash
# Create service account
gcloud iam service-accounts create flood-llm-sa \
  --display-name="FloodLLM Service Account"

# Grant Earth Engine access
gcloud projects add-iam-policy-binding flood-llm-PROJECT_ID \
  --member="serviceAccount:flood-llm-sa@flood-llm-PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/earthengine.admin"

# Generate key file
gcloud iam service-accounts keys create ee-key.json \
  --iam-account=flood-llm-sa@flood-llm-PROJECT_ID.iam.gserviceaccount.com
```

---

## 3. Containerization

### 3.1 Create Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ee-key.json

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 Build and Push Image

```bash
# Set variables
PROJECT_ID="flood-llm-PROJECT_ID"
REGION="us-central1"
REPO="flood-llm"
IMAGE="flood-llm-api"
TAG="latest"

# Create Artifact Registry repository
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="FloodLLM container images"

# Build and push
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG
```

---

## 4. Cloud Run Deployment

### 4.1 Deploy to Cloud Run

```bash
gcloud run deploy flood-llm \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=1 \
  --timeout=300 \
  --set-env-vars="GOOGLE_API_KEY=your-key,ENV=production" \
  --set-secrets="COPERNICUS_USERNAME=secret-name:latest,COPERNICUS_PASSWORD=secret-name:latest"
```

### 4.2 Configure Environment Variables

```bash
# Set non-secret variables
gcloud run services update flood-llm \
  --region=$REGION \
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,LOG_LEVEL=INFO"

# Store secrets in Secret Manager
echo -n "your-copernicus-username" | gcloud secrets create COPERNICUS_USERNAME --data-file=-
echo -n "your-copernicus-password" | gcloud secrets create COPERNICUS_PASSWORD --data-file=-
echo -n "your-google-api-key" | gcloud secrets create GOOGLE_API_KEY --data-file=-
```

### 4.3 Access the Service

```bash
# Get the service URL
gcloud run services describe flood-llm --region=$REGION --format='value(status.url)'

# Example output: https://flood-llm-xyz.a.run.app
```

---

## 5. Vertex AI Model Hosting (Optional)

If you want to use a custom fine-tuned model instead of Gemini:

### 5.1 Export Model

```python
# Save your fine-tuned model
import joblib
joblib.dump(model, "flood-model.pkl")
```

### 5.2 Upload to Cloud Storage

```bash
gsutil mb gs://flood-llm-models
gsutil cp flood-model.pkl gs://flood-llm-models/
```

### 5.3 Deploy to Vertex AI

```bash
gcloud ai models upload \
  --display-name="flood-llm-model" \
  --container-image-uri=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG

gcloud ai endpoints create \
  --display-name="flood-llm-endpoint"

gcloud ai endpoints deploy-model ENDPOINT_ID \
  --model=MODEL_ID \
  --display-name="flood-llm-deployment" \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=5
```

---

## 6. Production Configuration

### 6.1 Environment Variables

Create a `.env.production` file:

```bash
# API Keys
GOOGLE_API_KEY=your-google-api-key
COPERNICUS_USERNAME=your-copernicus-username
COPERNICUS_PASSWORD=your-copernicus-password
NASA_EARTHDATA_USERNAME=your-nasa-username
NASA_EARTHDATA_PASSWORD=your-nasa-password

# GCP
GCP_PROJECT_ID=flood-llm-PROJECT_ID
GCP_BUCKET_NAME=flood-llm-data

# App Settings
APP_ENV=production
LOG_LEVEL=WARNING

# Processing
DEFAULT_BUFFER_KM=50.0
WATER_THRESHOLD_VV=-17.0
CLOUD_COVER_MAX=20.0
```

### 6.2 Cloud Storage Bucket

```bash
# Create bucket for data storage
gsutil mb gs://flood-llm-data

# Set lifecycle policy (delete files older than 30 days)
gsutil lifecycle set lifecycle-policy.json gs://flood-llm-data
```

`lifecycle-policy.json`:
```json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
```

---

## 7. CI/CD Pipeline

### 7.1 Cloud Build Trigger

Create `cloudbuild.yaml`:

```yaml
steps:
  # Build container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$SHORT_SHA', '.']

  # Push to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$SHORT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'flood-llm'
      - '--image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$SHORT_SHA'
      - '--platform=managed'
      - '--region=$REGION'
      - '--allow-unauthenticated'

substitutions:
  _REGION: us-central1
  _REPO: flood-llm
  _IMAGE: flood-llm-api
```

### 7.2 Create Trigger

```bash
gcloud builds triggers create github \
  --repo="https://github.com/USERNAME/flood-llm" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"
```

---

## 8. Monitoring and Logging

### 8.1 Cloud Logging

All logs are automatically sent to Cloud Logging. View logs:

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=flood-llm" --limit=50
```

### 8.2 Cloud Monitoring Dashboard

```bash
# Create uptime check
gcloud monitoring uptime create flood-llm-uptime \
  --resource-type=cloud_run_service \
  --resource-labels=service_name=flood-llm,location=$REGION \
  --protocol=HTTPS \
  --path="/api/status" \
  --check-period=300
```

---

## 9. Security Considerations

### 9.1 Add Authentication

For production, add API key authentication:

```python
# app/api/main.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/prompt")
async def submit_prompt(request: PromptRequest, api_key: str = Depends(verify_api_key)):
    ...
```

### 9.2 CORS Configuration

Update CORS settings for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### 9.3 Rate Limiting

Add rate limiting with `slowapi`:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/prompt")
@limiter.limit("10/minute")
async def submit_prompt(request: PromptRequest):
    ...
```

---

## 10. Cost Optimization

| Resource | Estimated Cost (Monthly) | Optimization Tips |
|----------|--------------------------|-------------------|
| Cloud Run | $5-20 | Scale to 0 when idle |
| Earth Engine | Free (research tier) | Apply for grants |
| Cloud Storage | $1-5 | Lifecycle policies |
| Secret Manager | $0.50 | Cache secrets |
| Cloud Build | Free tier (120 min/month) | Use cached builds |

---

## 11. Troubleshooting

### "Service unavailable"

```bash
# Check service status
gcloud run services describe flood-llm --region=$REGION

# View recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=flood-llm" --limit=10
```

### "Earth Engine authentication failed"

Ensure the service account has Earth Engine enabled:

```bash
earthengine service_account --flood-llm-sa@flood-llm-PROJECT_ID.iam.gserviceaccount.com
```

### "Container build failed"

Check GDAL installation:

```dockerfile
# Use pre-built GDAL image
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.7
```

---

## Next Steps

- See [API Reference](API_REFERENCE.md) for endpoint documentation
- See [Architecture](ARCHITECTURE.md) for system design
- Set up custom domain with Cloud Run
