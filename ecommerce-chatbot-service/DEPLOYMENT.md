# FootyBot Deployment Guide

Complete guide for deploying the FootyBot chatbot service to production.

## 📋 Pre-Deployment Checklist

- [ ] FAISS index built (`retriever/vector_index.faiss` exists)
- [ ] Environment variables configured
- [ ] Dependencies tested locally
- [ ] API endpoints tested
- [ ] CORS origins configured
- [ ] Hugging Face API token obtained

## 🚀 Deployment Options

### Option 1: Render (Recommended)

**Pros:** Free tier available, automatic SSL, easy deployment
**Cons:** Cold starts on free tier

#### Steps:

1. **Prepare Repository**

   ```bash
   git add .
   git commit -m "Prepare chatbot for deployment"
   git push origin main
   ```

2. **Create Render Account**

   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create Web Service**

   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure Service**

   ```
   Name: footybot-chatbot
   Region: Choose nearest to your users
   Branch: main
   Root Directory: ml-service/ecommerce-chatbot-service
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app --workers 2 --threads 4 --timeout 120
   ```

5. **Set Environment Variables**

   ```
   HF_API_TOKEN=your_huggingface_token
   FRONTEND_URL=https://your-frontend.vercel.app
   FLASK_ENV=production
   ```

6. **Deploy**

   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Note the service URL (e.g., `https://footybot-xyz.onrender.com`)

7. **Update Frontend**
   - Update `NEXT_PUBLIC_CHATBOT_API` in your frontend `.env`:
     ```
     NEXT_PUBLIC_CHATBOT_API=https://footybot-xyz.onrender.com/chatbot
     ```

#### Post-Deployment

- Test health endpoint: `https://your-service.onrender.com/`
- Test chatbot endpoint with curl or Postman
- Monitor logs in Render dashboard

---

### Option 2: Heroku

**Pros:** Easy scaling, good documentation
**Cons:** No free tier (paid from $7/month)

#### Steps:

1. **Install Heroku CLI**

   ```bash
   brew install heroku/brew/heroku  # macOS
   ```

2. **Login**

   ```bash
   heroku login
   ```

3. **Create App**

   ```bash
   cd ml-service/ecommerce-chatbot-service
   heroku create footybot-chatbot
   ```

4. **Set Environment Variables**

   ```bash
   heroku config:set HF_API_TOKEN=your_token_here
   heroku config:set FRONTEND_URL=https://your-frontend.vercel.app
   ```

5. **Deploy**

   ```bash
   git push heroku main
   ```

6. **Scale**

   ```bash
   heroku ps:scale web=1
   ```

7. **View Logs**
   ```bash
   heroku logs --tail
   ```

---

### Option 3: Railway

**Pros:** Simple deployment, generous free tier
**Cons:** Less mature than Render

#### Steps:

1. **Create Railway Account**

   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Create New Project**

   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **Configure**

   - Root Directory: `ml-service/ecommerce-chatbot-service`
   - Start Command: `gunicorn app:app --workers 2 --threads 4 --timeout 120`

4. **Add Environment Variables**

   ```
   HF_API_TOKEN=your_token
   FRONTEND_URL=https://your-frontend.vercel.app
   ```

5. **Deploy**
   - Railway will auto-deploy on push to main

---

### Option 4: Google Cloud Run

**Pros:** Scales to zero, pay only for usage
**Cons:** More complex setup

#### Steps:

1. **Create Dockerfile**

   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 app:app
   ```

2. **Build and Push**

   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/footybot
   ```

3. **Deploy**
   ```bash
   gcloud run deploy footybot \
     --image gcr.io/PROJECT-ID/footybot \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars HF_API_TOKEN=your_token
   ```

---

## 🔧 Configuration Best Practices

### Environment Variables

**Required:**

- `HF_API_TOKEN`: Your Hugging Face API token
- `FRONTEND_URL`: Your frontend domain for CORS

**Optional:**

- `PORT`: Server port (auto-set by most platforms)
- `FLASK_ENV`: Set to `production`
- `WORKERS`: Number of Gunicorn workers (default: 2)

### Resource Allocation

**Minimum Requirements:**

- CPU: 1 vCPU
- RAM: 2 GB
- Disk: 1 GB

**Recommended for Production:**

- CPU: 2 vCPUs
- RAM: 4 GB
- Disk: 2 GB

### Gunicorn Configuration

```bash
gunicorn app:app \
  --workers 2 \           # 2-4 workers for most cases
  --threads 4 \           # 4 threads per worker
  --timeout 120 \         # 2 minute timeout for HF API
  --bind 0.0.0.0:$PORT \
  --log-level info \
  --access-logfile - \
  --error-logfile -
```

---

## 🧪 Testing Deployment

### 1. Health Check

```bash
curl https://your-service-url.com/
```

Expected response:

```json
{
  "status": "healthy",
  "service": "FootyBot Chatbot",
  "version": "2.0.0",
  "components": {
    "index_loaded": true,
    "chunks_loaded": true,
    "embedder_loaded": true
  }
}
```

### 2. Chatbot Test

```bash
curl -X POST https://your-service-url.com/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "What products do you sell?"}'
```

### 3. Frontend Integration

- Update frontend environment variable
- Test chatbot in browser
- Verify CORS works correctly

---

## 📊 Monitoring

### Metrics to Track

1. **Response Time**

   - Target: < 5 seconds average
   - Alert if: > 10 seconds

2. **Error Rate**

   - Target: < 1%
   - Alert if: > 5%

3. **Uptime**

   - Target: 99.5%+
   - Alert if: < 99%

4. **API Timeout Rate**
   - Target: < 5%
   - Alert if: > 10%

### Logging

**What to Log:**

- All incoming requests
- API errors and retries
- Cache hit/miss rates
- Response times
- User feedback

**Log Levels:**

- `INFO`: Normal operations
- `WARNING`: Retries, slow responses
- `ERROR`: Failed requests, API errors
- `CRITICAL`: Service initialization failures

---

## 🐛 Troubleshooting

### Issue: Service crashes on startup

**Symptoms:**

- Health check fails
- Logs show import errors

**Solutions:**

1. Check FAISS index exists: `retriever/vector_index.faiss`
2. Verify all dependencies installed
3. Check Python version (3.9+)
4. Review build logs

### Issue: Slow response times

**Symptoms:**

- Responses take > 10 seconds
- Timeout errors

**Solutions:**

1. Increase timeout in Gunicorn
2. Check HF API status
3. Add more workers/threads
4. Verify cache is working

### Issue: CORS errors in frontend

**Symptoms:**

- Browser console shows CORS error
- Preflight requests fail

**Solutions:**

1. Add frontend URL to `FRONTEND_URL` env var
2. Check CORS configuration in `app.py`
3. Verify protocol (http/https) matches

### Issue: Out of memory

**Symptoms:**

- Service crashes randomly
- 502/503 errors

**Solutions:**

1. Increase memory allocation
2. Reduce number of workers
3. Clear cache periodically
4. Use smaller embedding model

---

## 🔐 Security Checklist

- [ ] API token in environment variable (not hardcoded)
- [ ] CORS restricted to specific origins
- [ ] Input validation enabled
- [ ] Request size limits set
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] Logs don't contain sensitive data
- [ ] Error messages don't expose internals

---

## 📈 Scaling Guidelines

### When to Scale Up

**Indicators:**

- Response time > 5s consistently
- CPU usage > 80%
- Memory usage > 90%
- Queue depth increasing

**Actions:**

1. Increase workers/threads
2. Add more memory
3. Use multiple instances
4. Consider caching layer

### Cost Optimization

**Free Tier Strategy:**

- Use Render free tier (with cold starts)
- Scale to zero during low traffic
- Cache aggressively

**Production Strategy:**

- Start with minimal resources
- Scale based on metrics
- Use auto-scaling when available
- Consider serverless (Cloud Run)

---

## 🔄 CI/CD Setup

### GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Chatbot

on:
  push:
    branches: [main]
    paths:
      - "ml-service/ecommerce-chatbot-service/**"

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Deploy to Render
        env:
          RENDER_API_KEY: ${{ secrets.RENDER_API_KEY }}
        run: |
          curl -X POST https://api.render.com/deploy/YOUR_SERVICE_ID
```

---

## 📝 Post-Deployment Tasks

1. **Update Documentation**

   - [ ] Add production URL to README
   - [ ] Update API documentation
   - [ ] Document known issues

2. **Set Up Monitoring**

   - [ ] Configure uptime monitoring (UptimeRobot, Pingdom)
   - [ ] Set up error tracking (Sentry)
   - [ ] Enable log aggregation

3. **Test Everything**

   - [ ] Run full test suite
   - [ ] Test from different locations
   - [ ] Verify with real users

4. **Create Backup Plan**
   - [ ] Document rollback procedure
   - [ ] Keep previous version accessible
   - [ ] Test disaster recovery

---

## 🆘 Support Resources

- **Render Docs**: https://render.com/docs
- **Heroku Docs**: https://devcenter.heroku.com
- **Gunicorn Docs**: https://docs.gunicorn.org
- **FAISS Wiki**: https://github.com/facebookresearch/faiss/wiki

---

**Good luck with your deployment! 🚀**
