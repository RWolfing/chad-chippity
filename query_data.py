"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.memory import ConversationBufferMemory

def get_chain(
    model_name: str, vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
        model_name=model_name,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )

    doc_chain = load_qa_with_sources_chain(streaming_llm, chain_type="stuff", callback_manager=manager)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        memory=memory,
    )

    return qa
