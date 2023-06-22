"""Load html from files, clean up, split, ingest into Weaviate."""
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv
import os;

from langchain.document_loaders import ReadTheDocsLoader

load_dotenv()

chromaDir = './chroma/catalogue'

collectionName = "catalogue_collection"
model_name = os.getenv("OPENAI_MODEL_NAME")

def ingest_docs():
    ### Load and process data frame data frame
    loader = ReadTheDocsLoader("rtdocs", features="html.parser")
    documents = loader.load()

    embeddings = OpenAIEmbeddings(model=model_name)
    vectorstore = Chroma.from_documents(
        collection_name=collectionName,
        documents=documents,
        embedding=embeddings,
        persist_directory=chromaDir,
    )

    vectorstore.persist()
    print(f'Survey embeddings created for {len(documents)} documents')


if __name__ == "__main__":
    print(f'Running ingest with {model_name}...')
    ingest_docs()
