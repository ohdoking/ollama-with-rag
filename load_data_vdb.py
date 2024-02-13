import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

# Import the environment variables
import env_variables


def create_vector_db():
    loader = PyPDFDirectoryLoader(os.environ['DATA_PATH'])
    documents = loader.load()
    print(f"Processed {len(documents)} pdf files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vectorStore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), persist_directory=os.environ['DB_PATH'])
    vectorStore.persist()


if __name__ == "__main__":
    create_vector_db()
