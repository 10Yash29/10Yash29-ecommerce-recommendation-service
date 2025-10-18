#!/bin/bash

# FootyBot Chatbot Setup Script
# This script sets up the chatbot service for local development

set -e

echo "🤖 FootyBot Chatbot Setup"
echo "=========================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
echo "✅ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your HF_API_TOKEN"
    echo ""
fi

# Check if FAISS index exists
if [ ! -f "retriever/vector_index.faiss" ]; then
    echo "🔨 Building FAISS index..."
    python3 build_index.py
    echo "✅ FAISS index built"
else
    echo "✅ FAISS index already exists"
fi
echo ""

# Test import
echo "🧪 Testing imports..."
python3 -c "import faiss, sentence_transformers, flask" && echo "✅ All imports successful" || echo "❌ Import failed"
echo ""

echo "✨ Setup complete!"
echo ""
echo "To start the chatbot service:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the server: python3 app.py"
echo ""
echo "Or run with Gunicorn:"
echo "  gunicorn app:app --workers 2 --threads 4 --timeout 120"
echo ""
