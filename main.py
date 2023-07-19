"""Main entrypoint for the app."""
import logging
import os

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStore

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv

from callback import QuestionGenCallbackHandler
from query_data import get_chain
from schemas import ChatResponse


load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

CHROMA_DIR = './chroma/clutch_doc'
COLLECTION_NAME="clutch_doc"

@app.on_event("startup")
async def startup_event():
    if not Path(CHROMA_DIR).exists():
        raise ValueError("No vectorstore found. Please run ingest.py first.")
    
    global model_name
    model_name = os.getenv("OPENAI_CHAT_MODEL");

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL_NAME"))
    chromaClient = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))

    global vectorstore
    vectorstore = Chroma(collection_name=COLLECTION_NAME, client=chromaClient, embedding_function=embeddings)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(model_name, vectorstore, question_handler, tracing=False)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            
            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            
            resp = ChatResponse(sender="bot", message=result["answer"], type="stream")
            await websocket.send_json(resp.dict())

            #chat_history.append((question, "This is a token"))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
