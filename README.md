# ollama-with-rag

## Tech Stack
Chromadb — Vectorstore
gpt4all — text embeddings
langchain — Framework to facilitate Application Development using LLMs
chainlit — Build ChatGPT like interface

## Prerequisites
1. Install Ollama(https://ollama.com/download/mac)

2. Install python dependencies
```bash
  pip install -r requirements.txt
```

## How to use project

### Upload required Data and load into VectorStore
```bash
  python3 load_data_vdb.py
```

### How to run chatbot
```bash
  chainlit run bot.py -w
```
