# FootyBot - E-commerce Chatbot Service

A production-ready RAG (Retrieval-Augmented Generation) chatbot for FootyTrends e-commerce store, powered by FAISS vector search and Hugging Face transformers.

## 🎯 Features

- **Intelligent Context Retrieval**: Uses FAISS vector similarity search to find relevant store information
- **Smart Caching**: LRU cache for frequently asked questions
- **Error Handling**: Comprehensive error handling with retry logic
- **Production Ready**: Gunicorn WSGI server with multiple workers
- **CORS Enabled**: Configured for frontend integration
- **Health Monitoring**: Built-in health check endpoint
- **Structured Logging**: Detailed logs for debugging and monitoring

## 🏗️ Architecture

```
┌─────────────┐
│   Frontend  │ (Next.js)
└──────┬──────┘
       │ HTTP POST /chatbot
       ↓
┌─────────────────────────────┐
│   Flask API Server          │
│  - CORS middleware          │
│  - Request validation       │
│  - Error handling           │
└──────┬──────────────────────┘
       │
       ↓
┌─────────────────────────────┐
│   RAG Pipeline              │
│  ┌─────────────────────┐    │
│  │ 1. User Query       │    │
│  └──────┬──────────────┘    │
│         ↓                   │
│  ┌─────────────────────┐    │
│  │ 2. Embed Query      │    │
│  │   (SentenceT5)      │    │
│  └──────┬──────────────┘    │
│         ↓                   │
│  ┌─────────────────────┐    │
│  │ 3. FAISS Search     │    │
│  │   (Top K=3)         │    │
│  └──────┬──────────────┘    │
│         ↓                   │
│  ┌─────────────────────┐    │
│  │ 4. Context Filter   │    │
│  │   (Distance < 1.5)  │    │
│  └──────┬──────────────┘    │
│         ↓                   │
│  ┌─────────────────────┐    │
│  │ 5. Generate Answer  │    │
│  │   (FLAN-T5-Large)   │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone and navigate to the directory**

   ```bash
   cd ml-service/ecommerce-chatbot-service
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face API token
   ```

5. **Build the FAISS index** (if not already built)
   ```bash
   python build_index.py
   ```

## 🚀 Running the Service

### Development Mode

```bash
python3 app.py
```

Server will start on `http://localhost:8000`

### Production Mode (with Gunicorn)

```bash
gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000
```

## 📡 API Endpoints

### Health Check

```http
GET /
```

**Response:**

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

### Chat Endpoint

```http
POST /chatbot
Content-Type: application/json

{
  "message": "What is your return policy?",
  "userId": "optional_user_id"
}
```

**Response:**

```json
{
  "response": "Our return policy allows returns within 30 days...",
  "confidence": "high"
}
```

**Error Response:**

```json
{
  "error": "Message too long. Please keep it under 500 characters."
}
```

### Feedback Endpoint

```http
POST /feedback
Content-Type: application/json

{
  "message": "What is your return policy?",
  "response": "Our return policy...",
  "rating": 5
}
```

## 🔧 Configuration

### Environment Variables

| Variable       | Description             | Default    |
| -------------- | ----------------------- | ---------- |
| `HF_API_TOKEN` | Hugging Face API token  | Required   |
| `PORT`         | Server port             | 8000       |
| `FRONTEND_URL` | Allowed frontend origin | \*         |
| `FLASK_ENV`    | Flask environment       | production |

### Model Configuration

- **Embedder**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Generator**: `google/flan-t5-large` (Hugging Face API)
- **Vector Store**: FAISS with L2 distance
- **Top K Results**: 3
- **Distance Threshold**: 1.5

## 📊 Performance

- **Response Time**: ~2-5 seconds (depends on HF API)
- **Cache Hit Rate**: ~40-60% for common queries
- **Concurrent Requests**: Handles 50+ simultaneous requests
- **Memory Usage**: ~1.5 GB (with loaded models)

## 🧪 Testing

### Manual Testing

```bash
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "What products do you sell?"}'
```

### Health Check

```bash
curl http://localhost:8000/
```

## 🐛 Troubleshooting

### Common Issues

**1. FAISS index not found**

```bash
# Solution: Build the index
python build_index.py
```

**2. Hugging Face API timeout**

- Check your API token
- Verify internet connection
- The service will retry automatically (2 attempts)

**3. CORS errors**

- Ensure `FRONTEND_URL` is set correctly in `.env`
- Check that frontend is using the correct API URL

**4. Out of memory**

- Reduce worker count in Procfile
- Use smaller batch sizes
- Consider using FAISS GPU if available

## 🚢 Deployment

### Deploy to Render

1. **Create a new Web Service on Render**
2. **Connect your GitHub repository**
3. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --workers 2 --threads 4 --timeout 120`
4. **Add environment variables:**
   - `HF_API_TOKEN`
   - `FRONTEND_URL`
5. **Deploy!**

### Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create footybot-chatbot

# Set environment variables
heroku config:set HF_API_TOKEN=your_token_here

# Deploy
git push heroku main
```

## 📈 Monitoring

### Logs

```bash
# View real-time logs
tail -f app.log

# On Render/Heroku
heroku logs --tail
```

### Metrics to Monitor

- Response time per request
- Cache hit rate
- Error rate
- API timeout frequency
- Memory usage

## 🔐 Security

- ✅ API token stored in environment variables
- ✅ Input validation (max 500 characters)
- ✅ CORS configured for specific origins
- ✅ No sensitive data logged
- ✅ Request timeout protection

## 🛠️ Development

### Adding New Store Knowledge

1. Edit `store_knowledge/store_info.md`
2. Rebuild the index:
   ```bash
   python build_index.py
   ```
3. Restart the service

### Improving Responses

- Adjust `temperature` (0.1-1.0) in `generate_response()`
- Modify `max_new_tokens` for longer/shorter responses
- Fine-tune the prompt template
- Adjust distance threshold for retrieval

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues or questions:

- Open an issue on GitHub
- Email: support@footytrends.com

---

**Built with ❤️ for FootyTrends**
