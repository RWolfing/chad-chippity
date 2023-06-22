"""Load html from files, clean up, split, ingest into Weaviate."""
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv
import os;

load_dotenv()

directory = './data/'
chromaDir = './chroma/' 
chunkSize = 1100 
chunkOverlap = 200

collectionName = "my_collection"
model_name = os.getenv("OPENAI_MODEL_NAME")

def ingest_docs():
    dLoader = DirectoryLoader(directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_documents = dLoader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(model=model_name)
    vectorstore = Chroma.from_documents(
        collection_name=collectionName,
        documents=documents,
        embedding=embeddings,
        persist_directory=chromaDir,
    )

    vectorstore.persist()
    print(f'Embeddings created for {len(documents)} documents')


if __name__ == "__main__":
    print(f'Running ingest with {model_name}...')
    ingest_docs()
