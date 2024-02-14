# Ollama with RAG and Chainlit

This project is designed to use Ollama locally, run it with RAG (Retrieval-Augmented Generation), and use Chainlit for a UI chatbot.

## 🛠️ Tech Stack

- **Chromadb**: Used as a Vectorstore.
- **gpt4all**: Utilized for text embeddings.
- **langchain**: A framework that facilitates application development using LLMs (Language Learning Models).
- **chainlit**: Used to build a ChatGPT-like interface.
- **transformers**: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.
- **Torch**: Libraries that provide a wide range of algorithms for deep learning.
- **Peft**: A library for performance evaluation.

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:

1. Install Ollama. You can download it from the official website.
2. Install the necessary Python dependencies by running the following command in your terminal:
```bash
  pip install -r requirements.txt
```
3. create folder for data and vector store and model:
```bash
  mkdir data
  mkdir vectorstores/db
  mkdir model
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

### Run FineTunning with Lora
1. Update model name(`BASE_MODEL_NAME`) in `.env`
```
	BASE_MODEL_NAME=google/flan-t5-base
```
2. Run script
```bash
	python3 fine_tunning_with_lora.py
```

### Run Evaluation fine tunning model
1. Update fine-tuned model name(`FINE_TUNNING_MODEL_NAME`) in `.env`
```
	FINE_TUNNING_MODEL_NAME=model/peft-trained-model/flan-trained-*
```
2. Run script
```bash
	python3 evalution_lora_model.py
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📜 License
This project is licensed under the terms of the MIT license. For more details, see the LICENSE file.