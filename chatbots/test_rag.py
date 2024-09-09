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

# Set up parameters
dato = "2024-08-16" # For dp manual scraping version
bucket = "sjentoft"
prefix = f"db-files/dapla-manual/{dato}"


def set_key():
    # Check if the HUGGINGFACE_API_KEY is set
    api_key = os.getenv("HUGGINGFACE_API_KEY")

    if api_key:
            print("HUGGINGFACE_API_KEY is set.")
    else:
        os.environ["HUGGINGFACE_API_KEY"] = input("Enter your huggingface token: ")


def get_llm_new(model_path, local_path, need_token=False):
    """
    Function to get model from S3 bucket ond save locally for reading in.
    """
    if need_token:
        set_key() # in case model requires token

    llm = VLLM(
        model = model_path,
        max_model_len=8000, 
        #gpu_memory_utilization=0.5,
        trust_remote_code=True,
        #cache_dir = local_path
        download_dir=local_path
    )

    return llm

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
    # Set up vectorstore
    s = time.time()
    docs = create_db.split_html(bucket, prefix)
    vectorstore = create_db.create_vectorstore(docs)
    t = time.time() - s
    print(f"vectore store done in {t} seconds")

    s = time.time()
    model_path = "mistralai/Mistral-7B-v0.1"
    model_path = "openai-community/gpt2"
    local_path = "inference_model/gpt2"
    llm = get_llm_new(model_path, local_path)
    t = time.time() - s
    print(f"llm loaded in {t} seconds")

    # Get q's - replace with those from csv file. 
    qs_vec = ["What is Dapla?", "What is a PAT?"]

    prompt = get_prompt()

    rag_chain = (
        {"context": vectorstore.as_retriver(), "question": RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )
    
    s = time.time()

    answers = []
    for q in qs_vec:
        response = rag_chain.invoke(q)
        answers.append(response)
    print(answers)

    print(f'Response took: {time.time() - s} seconds')
