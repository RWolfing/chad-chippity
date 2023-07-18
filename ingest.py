# pylint: disable=E1136
# pylint: disable=E1137

import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import ReadTheDocsLoader

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = './chroma/catalogue'

COLLECTION_NAME = "catalogue_collection"
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

def ingest_docs():
    """Ingest content"""
    loader = ReadTheDocsLoader("rtdocs", features="html.parser")
    documents = loader.load()

    embeddings = OpenAIEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vectorstore.persist()
    print(f'Survey embeddings created for {len(documents)} documents')


if __name__ == "__main__":
    print(f'Running ingest with {MODEL_NAME}...')
    ingest_docs()
