"""Load html from files, clean up, split, ingest into Weaviate."""
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv
import os;

import pandas as pd
from google_play_scraper import Sort, reviews
from langchain.document_loaders.dataframe import DataFrameLoader

load_dotenv()

chromaDir = './chroma/survey'

collectionName = "survey_collection"
model_name = os.getenv("OPENAI_MODEL_NAME")

def ingest_docs():
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

    collectionName="survey_collection"

    ### Load and process data frame data frame
    loader = DataFrameLoader(df, page_content_column="content")
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
