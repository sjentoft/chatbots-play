# Simple case for testing a qa RAG bot

#from vllm import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import VLLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import transformers
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
#from langchain_community.llms import HuggingFacePipeline
#from langchain_huggingface import HuggingFacePipeline
#from langchain.memory import SimpleMemory
#import torch

import s3fs
import os
import time

import create_db
import os
import config
print(config.embeddings_model)


def set_key():
    # Check if the HUGGINGFACE_API_KEY is set
    api_key = os.getenv("HUGGINGFACE_API_KEY")

    if api_key:
            print("HUGGINGFACE_API_KEY is set.")
    else:
        os.environ["HUGGINGFACE_API_KEY"] = input("Enter your huggingface token: ")


def get_llm(model_path, local_path, need_token=False):
    """
    Function to get model from S3 bucket ond save locally for reading in.
    """
    if need_token:
        set_key() # in case model requires token

    model_config = {'max_new_tokesn' : 512,
                'temperature': 0.01}
    
    llm = CTransformers(
        config = model_config,
        model=config.inference_folder + config.inference_model,
    )   

    return llm


def get_chroma_db():
    embedding_function = HuggingFaceInstructEmbeddings(model_name = config.embeddings_model,
                                                model_kwargs={"device": "cuda"})
    db = Chroma(persist_directory=config.path_to_chromadb, embedding_function=embedding_function)
    return db


def get_template():
    template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.

    Context: {context}
    Question: {question}

    Helpful answer:
    """
    return template

def get_prompt():
    template = get_template()
    prompt = PromptTemplate(
        input_variables=["context", "human_input"], template=template
    )
    return prompt


if __name__ == "__main__":
    # Set vectorstore
    vectorstore = get_chroma_db()

    # Get llm
    llm = get_llm(model_path, local_path, True)

    # Get q's - replace with those from csv file. 
    qs_vec = ["What is Dapla?", "What is a PAT?"]

    prompt = get_prompt()

    rag_chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )

    answers = []
    for q in qs_vec:
        response = rag_chain.invoke(q)
        answers.append(response)
    print(answers)

