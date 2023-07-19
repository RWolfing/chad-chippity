"""Ingest content and create embeddings"""
import os
import glob
import git
import tiktoken

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders.merge import MergedDataLoader

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = './chroma/clutch_doc'

COLLECTION_NAME = "clutch_doc"
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


REPO_NAME = "clutch"
REPO_URL = "https://github.com/lyft/clutch.git"
LOCAL_PATH = f"./data/ignored/{REPO_NAME}"

def count_tokens(text, encoding):
    """Counts tokens of the given string"""
    return len(encoding.encode(text))

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
    
    enc = tiktoken.encoding_for_model(model_name=MODEL_NAME)
    
    def count_doc_tokens(doc):
        """Counts tokens of the given doc"""
        return count_tokens(doc.page_content, encoding=enc)
    
    token_count = list(map(count_doc_tokens, docs))
    token_total = sum(token_count);
    
    if token_total > 100000:
        print(f"Total token count is {token_total} which exceeds the limit of 1000000. Aborting...")
        
    embeddings = OpenAIEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    vectorstore.persist()
    
    print(f"Ingested {len(docs)} documents with a total of {token_total} tokens.")


if __name__ == "__main__":
    print(f'Running ingest with {MODEL_NAME}...')
    ingest_docs()
