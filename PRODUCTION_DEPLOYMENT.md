# Ollama Deep Researcher - Production Deployment Guide

This document outlines the production deployment configuration and steps for the Ollama Deep Researcher application.

## Production-Ready Features Implemented

1. **Containerization**
   - Multi-stage Docker build process
   - Optimized Docker image size with appropriate layers
   - `.dockerignore` file to exclude unnecessary files

2. **Environment Configuration**
   - Environment variables for all configurable settings
   - `.env.example` file with all required variables
   - Production vs development environment detection

3. **Frontend Optimization**
   - Next.js static export for optimal performance
   - Production build configuration
   - Asset optimization and caching headers

4. **Backend Improvements**
   - Production mode with disabled debugging
   - CORS properly configured for production
   - Error handling and logging improvements
   - Health check endpoint for monitoring

5. **Deployment Infrastructure**
   - Docker Compose for orchestration
   - Nginx configuration for production use
   - Volume mapping for persistent data

6. **Monitoring & Maintenance**
   - Health check script for monitoring
   - Logging configuration
   - Easy update process

## Deployment Methods

### 1. Quick Deployment (Recommended)

For a standard deployment with default settings:

```bash
./deploy.sh
```

This script will:
- Build the Next.js frontend
- Build the Docker containers for both the app and Ollama
- Start the application

### 2. Manual Deployment

For more control over the deployment process:

1. Build the Next.js frontend:
   ```bash
   cd frontend/next-ui
   npm ci
   npm run build
   cd ../..
   ```

2. Start the Docker containers:
   ```bash
   docker-compose up -d --build
   ```

### 3. Production Deployment with Nginx (Advanced)

For production environments with domain names and SSL:

1. Uncomment the nginx service in `docker-compose.yml`
2. Place your SSL certificates in the `./ssl` directory
3. Update the `nginx.conf` with your domain name
4. Deploy using Docker Compose:
   ```bash
   docker-compose up -d --build
   ```

## Configuration Options

All configuration is done through environment variables, which can be set in the `docker-compose.yml` file or a `.env` file.

Key configuration options:

- `PRODUCTION`: Set to "true" for production mode
- `OLLAMA_ENDPOINT`: URL for the Ollama API
- `LLM_MODEL`: The Ollama model to use
- `EMBEDDING_MODEL`: The embedding model to use

See `.env.example` for a complete list of configuration options.

## Monitoring and Maintenance

### Health Checks

Run the health check script to verify the application status:

```bash
./health-check.sh
```

### Viewing Logs

```bash
docker-compose logs -f
```

### Updating the Application

1. Pull the latest code:
   ```bash
   git pull
   ```

2. Rebuild and restart:
   ```bash
   ./deploy.sh
   ```

## Troubleshooting

### Common Issues

1. **Ollama connection errors**
   - Ensure Ollama is running and accessible
   - Check the Ollama logs: `docker-compose logs ollama`
   - Verify the `OLLAMA_ENDPOINT` environment variable

2. **WebSocket connection issues**
   - Check nginx configuration if using a reverse proxy
   - Ensure proper headers are set for WebSocket support

3. **Model loading failures**
   - Verify the model is available in Ollama
   - Check for sufficient disk space and GPU memory

For more detailed troubleshooting, check the application logs in the `logs/` directory. 