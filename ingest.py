"""Ingest content and create embeddings"""
import os
import glob
import git

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders.merge import MergedDataLoader

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = './chroma/survey'

COLLECTION_NAME = "survey_collection"
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


REPO_NAME = "clutch"
REPO_URL = "https://github.com/lyft/clutch.git"
LOCAL_PATH = f"./data/ignored/{REPO_NAME}"


def ingest_docs():
    """Ingest content"""
    path_doc = "docs/advanced/**/*.md"

    if not os.path.exists(LOCAL_PATH):
        git.Repo.clone_from(REPO_URL, LOCAL_PATH)
        print(f"Cloning {REPO_URL} to {LOCAL_PATH}")

    loaders = []
    for md_file_path in glob.glob(os.path.join(LOCAL_PATH, path_doc), recursive=True):
        loaders.append(UnstructuredMarkdownLoader(md_file_path))
    
    merged_loaders = MergedDataLoader(loaders=loaders)
    docs = merged_loaders.load()
    
    embeddings = OpenAIEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    vectorstore.persist()
    
    print(f"Loaded {len(docs)} documents")


if __name__ == "__main__":
    print(f'Running ingest with {MODEL_NAME}...')
    ingest_docs()
