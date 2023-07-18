# pylint: disable=E1136
# pylint: disable=E1137

"""Ingest content and create embeddings"""
import os
import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.dataframe import DataFrameLoader

from dotenv import load_dotenv
from google_play_scraper import Sort, reviews

load_dotenv()

CHROMA_DIR = './chroma/survey'

COLLECTION_NAME = "survey_collection"
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

def ingest_docs():
    """Ingest content"""
    result, continuation_token = reviews(
        'com.nianticlabs.pokemongo',
        lang='en', # defaults to 'en'
        country='us', # defaults to 'us'
        sort=Sort.NEWEST, # defaults to Sort.NEWEST
        count=500, # defaults to 100
        #filter_score_with=5 # defaults to None(means all score)
    )

    df = pd.DataFrame.from_dict(result)

    df = df.fillna("")
    df['at'] = df['at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['source'] = df['reviewId']

    ### Load and process data frame data frame
    loader = DataFrameLoader(df, page_content_column="content")
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
