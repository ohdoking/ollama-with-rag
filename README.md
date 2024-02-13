# Ollama with RAG and Chainlit

This project is designed to use Ollama locally, run it with RAG (Retrieval-Augmented Generation), and use Chainlit for a UI chatbot.

## 🛠️ Tech Stack

- **Chromadb**: Used as a Vectorstore.
- **gpt4all**: Utilized for text embeddings.
- **langchain**: A framework that facilitates application development using LLMs (Language Learning Models).
- **chainlit**: Used to build a ChatGPT-like interface.

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:

1. Install Ollama. You can download it from the official website.
2. Install the necessary Python dependencies by running the following command in your terminal:
```bash
  pip install -r requirements.txt
```

## 🚀 How to Use the Project
Follow these steps to get the project up and running:

### Upload Required Data and Load into VectorStore
Run the following command to load your data into the VectorStore:
```bash
  python3 load_data_vdb.py
```

### Run the Chatbot
You can start the chatbot by running the following command:
```bash
  chainlit run bot.py -w
```
This will start the chatbot with a web interface.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.