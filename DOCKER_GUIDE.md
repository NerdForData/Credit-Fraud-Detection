# Docker Deployment Guide

## 🐳 Docker Setup for Credit Risk Prediction System

This guide explains how to build and deploy the Credit Risk Prediction System using Docker.

---

## Prerequisites

- Docker installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (comes with Docker Desktop)

---

## Quick Start with Docker

### Option 1: Using Docker Compose (Recommended)

**1. Build and start the container:**
```bash
docker-compose up --build
```

**2. Access the application:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**3. Stop the container:**
```bash
docker-compose down
```

### Option 2: Using Docker CLI

**1. Build the image:**
```bash
docker build -t credit-risk-api:latest .
```

**2. Run the container:**
```bash
docker run -d \
  --name credit-risk-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/artifacts:/app/artifacts \
  credit-risk-api:latest
```

**3. View logs:**
```bash
docker logs -f credit-risk-api
```

**4. Stop the container:**
```bash
docker stop credit-risk-api
docker rm credit-risk-api
```

---

## Docker Configuration

### Dockerfile Overview

```dockerfile
FROM python:3.12-slim          # Base image
WORKDIR /app                   # Working directory
COPY requirements.txt .        # Copy dependencies
RUN pip install -r requirements.txt  # Install dependencies
COPY src/ models/ static/ ./   # Copy application files
EXPOSE 8000                    # Expose port
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Key Features

- ✅ **Multi-stage optimization** - Small image size
- ✅ **Health checks** - Auto-restart on failure
- ✅ **Volume mounts** - Persistent data
- ✅ **Hot reload** - Development mode support

---

## Docker Compose Configuration

### Services

**credit-risk-api:**
- Port: 8000
- Volumes: src/, models/, artifacts/, static/
- Network: credit-risk-network
- Restart: unless-stopped

### Environment Variables

You can add environment variables in `docker-compose.yml`:

```yaml
environment:
  - ENVIRONMENT=production
  - LOG_LEVEL=info
  - MODEL_PATH=models/pipeline.joblib
```

---

## Development vs Production

### Development Mode (Hot Reload)

```bash
docker-compose up
```

Changes to source files automatically reload the server.

### Production Mode

**Build optimized image:**
```bash
docker build -t credit-risk-api:production -f Dockerfile .
```

**Run in production:**
```bash
docker run -d \
  --name credit-risk-api-prod \
  -p 8000:8000 \
  --restart unless-stopped \
  credit-risk-api:production
```

---

## Docker Commands Cheatsheet

### Build & Run
```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -d -p 8000:8000 --name credit-risk-api credit-risk-api

# Run with volume mounts
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name credit-risk-api credit-risk-api
```

### Manage Containers
```bash
# List running containers
docker ps

# View logs
docker logs credit-risk-api
docker logs -f credit-risk-api  # Follow logs

# Stop container
docker stop credit-risk-api

# Remove container
docker rm credit-risk-api

# Restart container
docker restart credit-risk-api
```

### Images
```bash
# List images
docker images

# Remove image
docker rmi credit-risk-api

# Prune unused images
docker image prune
```

### Docker Compose
```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Execute command in container
docker-compose exec credit-risk-api bash
```

---

## Troubleshooting

### Issue: Port already in use
```bash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Use different port
docker run -p 8080:8000 credit-risk-api
```

### Issue: Container exits immediately
```bash
# Check logs
docker logs credit-risk-api

# Run interactively
docker run -it credit-risk-api bash
```

### Issue: Model files not found
```bash
# Ensure models directory exists and has files
ls models/

# Mount models directory
docker run -v $(pwd)/models:/app/models credit-risk-api
```

### Issue: Permission denied
```bash
# On Linux, fix permissions
sudo chown -R $USER:$USER ./models ./artifacts
```

---

## Cloud Deployment

### AWS ECS

**1. Push to ECR:**
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag credit-risk-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest
```

**2. Create ECS task definition and service**

### Google Cloud Run

```bash
# Build for Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/credit-risk-api

# Deploy
gcloud run deploy credit-risk-api \
  --image gcr.io/PROJECT-ID/credit-risk-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name credit-risk-rg --location eastus

# Deploy container
az container create \
  --resource-group credit-risk-rg \
  --name credit-risk-api \
  --image credit-risk-api:latest \
  --dns-name-label credit-risk-api \
  --ports 8000
```

---

## Performance Optimization

### Multi-stage Build

Create optimized production image:

```dockerfile
# Builder stage
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
COPY src/ models/ static/ ./
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Resource Limits

```yaml
# docker-compose.yml
services:
  credit-risk-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

---

## Security Best Practices

1. **Use specific base images**
   ```dockerfile
   FROM python:3.12.1-slim  # Instead of python:3.12
   ```

2. **Run as non-root user**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

3. **Scan for vulnerabilities**
   ```bash
   docker scan credit-risk-api:latest
   ```

4. **Use secrets for sensitive data**
   ```bash
   docker secret create model_key model.key
   ```

---

## Monitoring

### Health Checks

Built-in health check endpoint: `/health`

```bash
curl http://localhost:8000/health
```

### Container Stats

```bash
# Real-time stats
docker stats credit-risk-api

# Inspect container
docker inspect credit-risk-api
```

---

## Next Steps

- ✅ Docker setup complete
- 🔄 Set up CI/CD pipeline
- 🔄 Configure load balancer
- 🔄 Add monitoring (Prometheus/Grafana)
- 🔄 Implement logging aggregation
- 🔄 Set up auto-scaling

---

**For more information, see the main README.md**
